"""
01b_scrape_draft_class_only.py
================================
Targeted scraper for historical draft classes. Instead of scraping all ~347
NCAA teams per year, this script:

  1. Reads the WNBA draft history to get each pick's college + draft year.
  2. Maps college names to sports-reference team slugs using the existing
     team CSVs (which already have the URLs).
  3. Scrapes only the unique (college, year) team pages needed — typically
     ~60–80 pages for a 4-year range vs ~1,400 for a full re-scrape.
  4. Appends the results to the existing ncaaw_players_raw_{year}.csv files
     so the rest of the pipeline (steps 03–08) works unchanged.

Why include the full team roster, not just the drafted player?
  Scraping the full team page costs nothing extra (one HTTP request regardless),
  and having teammates' stats in the features CSV improves z-score context for
  the drafted player's advanced metrics (e.g. usg_pct, ast_pct are team-relative).

Usage:
    python pipeline/01b_scrape_draft_class_only.py
    python pipeline/01b_scrape_draft_class_only.py --years 2018 2019 2020 2021
    python pipeline/01b_scrape_draft_class_only.py --force   # re-scrape even if file exists

Inputs:
    data/raw/wnba_draft_history.csv           (from 07_xgboost_model.py)
    data/raw/ncaaw_teams_{year}.csv           (any year — used for slug mapping)

Outputs:
    data/raw/ncaaw_players_raw_{year}.csv     (appended or created per year)
"""

import re
import time
import random
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"

# Years to scrape if --years not specified.
# These are the college season years (= draft year, since SR uses end-of-season year).
DEFAULT_YEARS = [2018, 2019, 2020, 2021]

MAX_RETRIES  = 5
SLEEP_RANGE  = (4.0, 7.0)
MIN_SIM      = 0.72   # minimum fuzzy-match score to accept a college name match

# Sports-reference base
SR_BASE = "https://www.sports-reference.com"

# Per-game and advanced table IDs (mirrors step 01)
PER_GAME_IDS = ["players_per_game", "per_game"]
ADV_IDS      = ["players_advanced", "advanced"]
TOTALS_RE    = re.compile(r"team totals|totals", re.IGNORECASE)

# Known college name aliases (draft history name → canonical team name in SR)
COLLEGE_ALIASES = {
    "uconn":              "Connecticut",
    "unc":                "North Carolina",
    "usc":                "USC",
    "lsu":                "Louisiana State",
    "smu":                "SMU",
    "tcu":                "TCU",
    "vcu":                "VCU",
    "utep":               "UTEP",
    "utsa":               "UTSA",
    "unlv":               "UNLV",
    "ucsb":               "UC Santa Barbara",
    "ucsd":               "UC San Diego",
    "uci":                "UC Irvine",
    "uab":                "UAB",
    "fiu":                "Florida International",
    "fau":                "Florida Atlantic",
    "ole miss":           "Mississippi",
    "penn":               "Pennsylvania",
    "miami":              "Miami (FL)",
    "miami (oh)":         "Miami (OH)",
    "pitt":               "Pittsburgh",
    "a&m":                "Texas A&M",
    "texas a&m":          "Texas A&M",
    "nc state":           "NC State",
    "ohio":               "Ohio",
    "ohio st":            "Ohio State",
    "ohio state":         "Ohio State",
    "arizona st":         "Arizona State",
    "michigan st":        "Michigan State",
    "penn st":            "Penn State",
    "penn state":         "Penn State",
    "iowa st":            "Iowa State",
    "washington st":      "Washington State",
    "kansas st":          "Kansas State",
    "oklahoma st":        "Oklahoma State",
    "oregon st":          "Oregon State",
    "utah st":            "Utah State",
    "colorado st":        "Colorado State",
    "san diego st":       "San Diego State",
    "boise st":           "Boise State",
    "fresno st":          "Fresno State",
    "cal":                "California",
    "cal poly":           "Cal Poly",
    "little rock":        "Arkansas-Little Rock",
    "sfa":                "Stephen F. Austin",
    "stephen f. austin":  "Stephen F. Austin",
    "south florida":      "USF",
    "saint joseph's":     "Saint Joseph's",
    "st. joseph's":       "Saint Joseph's",
    "st. john's":         "St. John's (NY)",
    "saint mary's":       "Saint Mary's (CA)",
    "mount st. mary's":   "Mount St. Mary's",
    "loyola chicago":     "Loyola (IL)",
    "loyola (md)":        "Loyola (MD)",
    "loyola marymount":   "LMU (CA)",
    "lmu":                "LMU (CA)",
    "rutgers university":       "Rutgers",
    "mercer university":        "Mercer",
    "rice university":                    "Rice",
    "james madison university":           "James Madison",
    "rider university":                   "Rider",
    "university of the pacific":          "Pacific",
    "university of tennessee at martin":  "UT Martin",
    "lafayette college":                  "Lafayette",
}

# ---------------------------------------------------------------------------
# HTTP helpers (mirrors step 01)
# ---------------------------------------------------------------------------

def fetch_html(url: str, session: requests.Session) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed {url}: {e}")
            time.sleep(10.0 * attempt)
    return ""


def soups_with_comments(html: str):
    soup = BeautifulSoup(html, "html.parser")
    result = [soup]
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            result.append(BeautifulSoup(c, "html.parser"))
    return result


def _find_table(soups, ids):
    for tid in ids:
        for s in soups:
            t = s.find("table", id=tid)
            if t:
                return t, tid
    return None, None


def _parse_player_cell(tr):
    for stat in ("name_display", "player"):
        td = tr.find("td", attrs={"data-stat": stat})
        if td:
            name = td.get_text(strip=True) or None
            a    = td.find("a", href=True)
            href = a["href"].strip() if a else None
            pid  = href.split("/")[-1].replace(".html", "") if href else None
            if name:
                return name, href, pid
    th = tr.find("th")
    if not th:
        return None, None, None
    name = th.get_text(strip=True) or None
    a    = th.find("a", href=True)
    href = a["href"].strip() if a else None
    pid  = href.split("/")[-1].replace(".html", "") if href else None
    return name, href, pid


def _table_to_df(table) -> pd.DataFrame:
    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()
    rows = []
    for tr in tbody.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        name, href, pid = _parse_player_cell(tr)
        if not name or TOTALS_RE.search(name):
            continue
        row = {"player": name, "player_href": href, "player_id": pid}
        for td in tr.find_all("td"):
            ds = td.get("data-stat")
            if ds:
                row[ds] = td.get_text(strip=True)
        rows.append(row)
    return pd.DataFrame(rows)


def _coalesce_xy(df: pd.DataFrame) -> pd.DataFrame:
    for col_x in [c for c in df.columns if c.endswith("_x")]:
        base  = col_x[:-2]
        col_y = base + "_y"
        if col_y in df.columns:
            df[base] = df[col_x].combine_first(df[col_y])
            df = df.drop(columns=[col_x, col_y])
    return df


def scrape_team_page(team_url: str, session: requests.Session) -> pd.DataFrame:
    """Scrape per-game + advanced tables from one team page, return merged df."""
    soups   = soups_with_comments(fetch_html(team_url, session))
    per_tbl, per_id = _find_table(soups, PER_GAME_IDS)
    adv_tbl, adv_id = _find_table(soups, ADV_IDS)

    if per_tbl is None and adv_tbl is None:
        raise RuntimeError("No player tables found")

    per_df = _table_to_df(per_tbl) if per_tbl else pd.DataFrame()
    adv_df = _table_to_df(adv_tbl) if adv_tbl else pd.DataFrame()

    if per_df.empty:
        return adv_df
    if adv_df.empty:
        return per_df

    id_cols  = {"player", "player_id", "player_href"}
    join_key = "player_id" if per_df["player_id"].notna().any() else "player"

    per_df = per_df.rename(columns={c: f"pg_{c}" for c in per_df.columns if c not in id_cols})
    adv_df = adv_df.rename(columns={c: f"adv_{c}" for c in adv_df.columns if c not in id_cols})

    merged = pd.merge(per_df, adv_df, on=join_key, how="outer")
    merged = _coalesce_xy(merged)
    merged["per_game_table_id"] = per_id
    merged["advanced_table_id"] = adv_id
    return merged

# ---------------------------------------------------------------------------
# College name → team URL mapping
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def build_college_url_map() -> dict:
    """
    Load all available ncaaw_teams_{year}.csv files and build a mapping:
        normalised_team_name → (team_name, team_url_template)
    where team_url_template has {year} as a placeholder.

    Uses the most recent year's CSV for each team so the mapping reflects
    current school names (slugs never change on SR).
    """
    team_csvs = sorted(RAW_DIR.glob("ncaaw_teams_*.csv"), reverse=True)
    if not team_csvs:
        raise FileNotFoundError(
            "No ncaaw_teams_{year}.csv files found in data/raw/. "
            "Run 01_scrape_multi_year.py first for at least one year."
        )

    mapping = {}  # norm_name → (display_name, url_template)
    for csv in team_csvs:
        df = pd.read_csv(csv)
        if "team" not in df.columns or "team_url" not in df.columns:
            continue
        for _, row in df.iterrows():
            name     = str(row["team"]).strip()
            url      = str(row["team_url"]).strip()
            norm     = _normalise(name)
            if norm and norm not in mapping:
                # Replace the year number in the URL with a placeholder
                m = re.search(r"/(\d{4})\.html$", url)
                if m:
                    template = url.replace(f"/{m.group(1)}.html", "/{year}.html")
                    mapping[norm] = (name, template)

    print(f"  Built team URL map: {len(mapping)} teams from {len(team_csvs)} team CSV(s)")
    return mapping


def resolve_college(college_name: str, url_map: dict, year: int) -> tuple:
    """
    Map a draft history college name to a sports-reference team URL for the
    given year. Returns (team_name, url) or (None, None) if not found.
    """
    raw_norm = _normalise(college_name)

    # Check alias table first
    alias_key = raw_norm
    for k, v in COLLEGE_ALIASES.items():
        if _normalise(k) == raw_norm:
            alias_key = _normalise(v)
            break

    # Exact match
    if alias_key in url_map:
        name, tmpl = url_map[alias_key]
        return name, tmpl.format(year=year)

    # Fuzzy match
    best_sim  = 0.0
    best_key  = None
    for norm_key in url_map:
        sim = _similarity(alias_key, norm_key)
        if sim > best_sim:
            best_sim = sim
            best_key = norm_key

    if best_sim >= MIN_SIM and best_key:
        name, tmpl = url_map[best_key]
        return name, tmpl.format(year=year)

    return None, None

# ---------------------------------------------------------------------------
# Main scraping logic
# ---------------------------------------------------------------------------

def load_draft_history() -> pd.DataFrame:
    path = RAW_DIR / "wnba_draft_history.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run pipeline/07_xgboost_model.py first to scrape "
            "the WNBA draft history."
        )
    return pd.read_csv(path)


def scrape_year(year: int, picks: pd.DataFrame, url_map: dict,
                session: requests.Session, force: bool) -> pd.DataFrame:
    """
    Scrape all unique colleges represented in `picks` for the given year.
    Returns a DataFrame in the same format as ncaaw_players_raw_{year}.csv.
    """
    out_path = RAW_DIR / f"ncaaw_players_raw_{year}.csv"

    # Collect unique colleges for this year
    unique_colleges = picks["college"].dropna().unique()

    # Resolve each college to a URL
    resolved = {}  # team_url → team_name
    unresolved = []
    for college in unique_colleges:
        team_name, url = resolve_college(college, url_map, year)
        if url:
            resolved[url] = team_name
        else:
            unresolved.append(college)

    if unresolved:
        print(f"  [WARN] Could not resolve {len(unresolved)} colleges: {unresolved}")

    # Short-circuit: if the raw file already exists and force is not set, skip.
    # Historical years (2018–2021) never gain new players, so re-scraping wastes
    # time and rate-limit budget. Use --force to override.
    if out_path.exists() and not force:
        existing = pd.read_csv(out_path, low_memory=False)
        if len(existing) > 0:
            print(f"  {out_path.name} already exists ({len(existing)} rows) — skipping. "
                  "Use --force to re-scrape.")
            return existing

    print(f"  Scraping {len(resolved)} team pages for {year} "
          f"({len(picks)} draft picks, {len(unique_colleges)} unique colleges)...")

    all_rows = []
    for i, (url, team_name) in enumerate(resolved.items(), 1):
        try:
            df_team = scrape_team_page(url, session)
            if df_team.empty:
                print(f"    [{i}/{len(resolved)}] {team_name}: 0 rows")
                continue

            # Normalize pg_/adv_ prefixes (mirrors step 01)
            col_map = {}
            id_cols = {"player", "player_id", "player_href",
                       "per_game_table_id", "advanced_table_id"}
            for col in df_team.columns:
                if col in id_cols:
                    continue
                if col.startswith("pg_"):
                    col_map[col] = col[3:]
                elif col.startswith("adv_"):
                    base = col[4:]
                    if base not in col_map.values():
                        col_map[col] = base
            df_team = df_team.rename(columns=col_map)

            # Add context columns
            df_team["season_year"] = year
            df_team["team"]        = team_name

            # Extract team_id and conference from the URL / existing team CSV
            m = re.search(r"/cbb/schools/([^/]+)/", url)
            df_team["team_id"] = m.group(1) if m else ""

            # Try to get conference from the url_map entry
            df_team["conference"] = ""
            df_team["team_url"]   = url

            # Coerce numerics
            skip = {"player", "player_id", "player_href", "team", "conference",
                    "conference_slug", "team_url", "team_id",
                    "per_game_table_id", "advanced_table_id", "awards", "adv_awards"}
            for c in df_team.columns:
                if c not in skip:
                    converted = pd.to_numeric(df_team[c], errors="coerce")
                    if converted.notna().sum() > 0:
                        df_team[c] = converted

            all_rows.append(df_team)
            print(f"    [{i}/{len(resolved)}] {team_name}: {len(df_team)} players")

        except Exception as e:
            print(f"    [{i}/{len(resolved)}] [FAIL] {team_name}: {e}")

        time.sleep(random.uniform(*SLEEP_RANGE))

    if not all_rows:
        return pd.DataFrame()

    new_df = pd.concat(all_rows, ignore_index=True)

    # Fill conference from team CSVs where possible
    team_csv = RAW_DIR / f"ncaaw_teams_{year}.csv"
    if not team_csv.exists():
        # Use any available year — conference membership changes rarely
        fallbacks = sorted(RAW_DIR.glob("ncaaw_teams_*.csv"))
        team_csv  = fallbacks[-1] if fallbacks else None

    if team_csv and team_csv.exists():
        teams_df = pd.read_csv(team_csv)
        if "team" in teams_df.columns and "conference" in teams_df.columns:
            conf_map = dict(zip(teams_df["team"], teams_df["conference"]))
            new_df["conference"] = new_df["team"].map(conf_map).fillna("")

    # Merge with existing file (avoid duplicate players)
    if out_path.exists() and not force:
        existing = pd.read_csv(out_path, low_memory=False)
        # Identify which players from new_df are already in existing
        if "player_id" in existing.columns and "player_id" in new_df.columns:
            existing_ids = set(existing["player_id"].dropna().astype(str))
            new_df = new_df[
                ~new_df["player_id"].astype(str).isin(existing_ids)
            ]
        elif "player" in existing.columns:
            existing_names = set(existing["player"].dropna().str.strip().str.lower())
            new_df = new_df[
                ~new_df["player"].str.strip().str.lower().isin(existing_names)
            ]

        if new_df.empty:
            print(f"  No new players to add for {year} (all already in {out_path.name})")
            return existing

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"  Appended {len(new_df)} new players → {out_path.name} "
              f"(total: {len(combined)})")
        return combined

    else:
        new_df.to_csv(out_path, index=False)
        action = "Overwrote" if (out_path.exists() and force) else "Created"
        print(f"  {action} {out_path.name} ({len(new_df)} players)")
        return new_df


def main():
    parser = argparse.ArgumentParser(
        description="Targeted scraper: only fetches colleges of WNBA draft picks."
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=DEFAULT_YEARS,
        help=f"College season years to scrape (default: {DEFAULT_YEARS})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing raw CSVs instead of appending"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 01b — Targeted Draft Class Scraper")
    print("=" * 60)
    print(f"  Years: {args.years}")

    draft_df = load_draft_history()
    print(f"  Loaded {len(draft_df)} draft picks "
          f"({draft_df['draft_year'].min()}–{draft_df['draft_year'].max()})")

    url_map = build_college_url_map()

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.sports-reference.com/",
        })

        for year in sorted(args.years):
            print(f"\n{'─'*50}")
            print(f"  Year: {year}")

            # Draft picks for this year (college season year = draft year)
            year_picks = draft_df[draft_df["draft_year"] == year].copy()
            if year_picks.empty:
                print(f"  No draft picks found for {year} in draft history")
                continue

            scrape_year(year, year_picks, url_map, session, args.force)

    print("\n" + "=" * 60)
    print("Done. Re-run from step 03 to incorporate new data:")
    print("  python run_pipeline.py --scrape --from-step 3")
    print("=" * 60)


if __name__ == "__main__":
    main()
