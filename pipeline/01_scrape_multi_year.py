"""
NCAAW Multi-Year Scraper
========================
Scrapes team standings and per-game + advanced player stats for multiple
seasons from sports-reference.com.

Usage:
    python pipeline/01_scrape_multi_year.py

Outputs (in data/raw/):
    ncaaw_teams_{YEAR}.csv      — team standings per year
    ncaaw_players_raw_{YEAR}.csv — player stats per year
"""

import re
import time
import random
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
YEARS = [2022, 2023, 2024, 2025, 2026]
OUT_DIR = Path(__file__).parent.parent / "data" / "raw"
BASE = "https://www.sports-reference.com"

CONFERENCES = [
    ("Southeastern Conference", "sec"),
    ("Big Ten Conference", "big-ten"),
    ("Missouri Valley Conference", "mvc"),
    ("Atlantic Coast Conference", "acc"),
    ("West Coast Conference", "wcc"),
    ("Pac-12 Conference", "pac-12"),          # disbanded after 2023-24; needed for historical data
    ("Atlantic 10 Conference", "atlantic-10"),
    ("Big 12 Conference", "big-12"),
    ("Big East Conference", "big-east"),
    ("Horizon League", "horizon"),
    ("America East Conference", "america-east"),
    ("Western Athletic Conference", "wac"),
    ("Mountain West Conference", "mwc"),
    ("Coastal Athletic Association", "coastal"),
    ("American Athletic Conference", "aac"),
    ("Ivy League", "ivy"),
    ("Southern Conference", "southern"),
    ("Big Sky Conference", "big-sky"),
    ("Mid-Eastern Athletic Conference", "meac"),
    ("Sun Belt Conference", "sun-belt"),
    ("Summit League", "summit"),
    ("Southwest Athletic Conference", "swac"),
    ("Ohio Valley Conference", "ovc"),
    ("Conference USA", "cusa"),
    ("Big West Conference", "big-west"),
    ("Atlantic Sun Conference", "atlantic-sun"),
    ("Mid-American Conference", "mac"),
    ("Big South Conference", "big-south"),
    ("Metro Atlantic Athletic Conference", "maac"),
    ("Southland Conference", "southland"),
    ("Northeast Conference", "northeast"),
    ("Patriot League", "patriot"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.sports-reference.com/",
}

TEAM_HREF_PATTERNS = [
    re.compile(r"^/cbb/schools/[^/]+/women/\d{4}\.html$"),
    re.compile(r"^/cbb/schools/[^/]+/women/?$"),
    re.compile(r"^/cbb/schools/[^/]+/women\.html$"),
]

MAX_RETRIES = 5
SLEEP_TEAMS = (4.0, 7.0)
SLEEP_CONF  = 4.0

DRAFT_YEAR           = max(YEARS)
FIRST_SEASONS_CACHE  = OUT_DIR / "player_first_seasons.csv"

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_html(url: str, session: requests.Session) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(10.0 * attempt)
    raise RuntimeError(f"Failed {url}: {last_err}")


def soups_with_comments(html: str):
    """Sports-Reference hides tables in HTML comments. Return all parseable soups."""
    soup = BeautifulSoup(html, "html.parser")
    soups = [soup]
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            soups.append(BeautifulSoup(c, "html.parser"))
    return soups

# ---------------------------------------------------------------------------
# Team standings scraper
# ---------------------------------------------------------------------------

def _find_standings_table(soups):
    for s in soups:
        t = s.find("table", id="standings")
        if t:
            return t
    for s in soups:
        for table in s.find_all("table"):
            thead = table.find("thead")
            if not thead:
                continue
            hs = {th.get_text(strip=True) for th in thead.find_all("th")}
            if {"W", "L", "W-L%"}.issubset(hs) and ("School" in hs or "Team" in hs):
                return table
    return None


def _cell(tr, data_stat: str):
    td = tr.find(["td", "th"], attrs={"data-stat": data_stat})
    return td.get_text(strip=True) if td else None


def scrape_conference_teams(conf_name, conf_slug, year, session):
    url = f"{BASE}/cbb/conferences/{conf_slug}/women/{year}.html"
    soups = soups_with_comments(fetch_html(url, session))
    table = _find_standings_table(soups)
    if not table:
        raise RuntimeError(f"No standings table at {url}")

    teams, seen = [], set()
    tbody = table.find("tbody")
    if not tbody:
        return teams

    for tr in tbody.find_all("tr"):
        cls = tr.get("class") or []
        if "thead" in cls or "spacer" in cls:
            continue
        a = tr.find("a", href=True)
        if not a:
            continue
        href = a["href"].strip()
        if not any(p.match(href) for p in TEAM_HREF_PATTERNS):
            continue

        team_id = href.split("/")[3]
        key = (conf_slug, year, team_id)
        if key in seen:
            continue
        seen.add(key)

        wins   = _cell(tr, "wins")   or _cell(tr, "w")
        losses = _cell(tr, "losses") or _cell(tr, "l")
        wl_pct = _cell(tr, "win_loss_pct") or _cell(tr, "win_loss_perc")

        teams.append({
            "season_year":    year,
            "conference":     conf_name,
            "conference_slug": conf_slug,
            "team":           a.get_text(strip=True),
            "team_id":        team_id,
            "wins":           int(wins)   if wins   and wins.isdigit()   else wins,
            "losses":         int(losses) if losses and losses.isdigit() else losses,
            "wl_pct":         float(wl_pct) if wl_pct and re.fullmatch(r"\d*\.\d+", wl_pct) else wl_pct,
            "team_url":       urljoin(BASE, href),
        })
    return teams


def scrape_teams_for_year(year: int, session: requests.Session) -> pd.DataFrame:
    rows = []
    for conf_name, conf_slug in CONFERENCES:
        try:
            r = scrape_conference_teams(conf_name, conf_slug, year, session)
            rows.extend(r)
            print(f"  [OK] {conf_name}: {len(r)} teams")
        except Exception as e:
            print(f"  [FAIL] {conf_name} ({conf_slug}): {e}")
        time.sleep(SLEEP_CONF)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.drop_duplicates(subset=["season_year", "team_id"])
              .sort_values(["conference", "team"])
              .reset_index(drop=True))

# ---------------------------------------------------------------------------
# Player stats scraper
# ---------------------------------------------------------------------------

PER_GAME_IDS = ["players_per_game", "per_game"]
ADV_IDS      = ["players_advanced", "advanced"]
TOTALS_RE    = re.compile(r"team totals|totals", re.IGNORECASE)


def _find_table(soups, ids):
    for tid in ids:
        for s in soups:
            t = s.find("table", id=tid)
            if t:
                return t, tid
    return None, None


def _parse_player_cell(tr):
    """
    Extract (name, href, player_id) from a table row.
    Sports-Reference moved player names/links from <th> into
    <td data-stat="name_display"> in their updated table format,
    so we check that first before falling back to <th>.
    """
    # New SR format: player name + link in <td data-stat="name_display">
    for stat in ("name_display", "player"):
        td = tr.find("td", attrs={"data-stat": stat})
        if td:
            name = td.get_text(strip=True) or None
            a = td.find("a", href=True)
            href = a["href"].strip() if a else None
            pid  = href.split("/")[-1].replace(".html", "") if href else None
            if name:
                return name, href, pid

    # Legacy fallback: name in <th>
    th = tr.find("th")
    if not th:
        return None, None, None
    name = th.get_text(strip=True) or None
    a = th.find("a", href=True)
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
    """
    After an outer merge, pandas adds _x/_y suffixes when both sides share a
    column that wasn't the join key. Coalesce each pair back to a single column.
    """
    x_cols = [c for c in df.columns if c.endswith("_x")]
    for col_x in x_cols:
        base = col_x[:-2]
        col_y = base + "_y"
        if col_y in df.columns:
            df[base] = df[col_x].combine_first(df[col_y])
            df = df.drop(columns=[col_x, col_y])
    return df


def scrape_team_players(team_url, session) -> pd.DataFrame:
    soups    = soups_with_comments(fetch_html(team_url, session))
    per_tbl, per_id = _find_table(soups, PER_GAME_IDS)
    adv_tbl, adv_id = _find_table(soups, ADV_IDS)

    if per_tbl is None and adv_tbl is None:
        raise RuntimeError("No player tables found")

    per_df = _table_to_df(per_tbl) if per_tbl is not None else pd.DataFrame()
    adv_df = _table_to_df(adv_tbl) if adv_tbl is not None else pd.DataFrame()

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


def _team_id_from_url(url):
    try:
        parts = urlparse(url).path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "cbb" and parts[1] == "schools":
            return parts[2]
    except Exception:
        pass
    return None


def scrape_players_for_year(teams_df: pd.DataFrame) -> pd.DataFrame:
    context_cols = [c for c in teams_df.columns if c != "team_url"]
    all_players  = []

    with requests.Session() as session:
        for i, row in teams_df.iterrows():
            team_url  = row["team_url"]
            team_name = row.get("team", f"row_{i}")
            try:
                df_p = scrape_team_players(team_url, session)
                if df_p.empty:
                    print(f"    [WARN] {team_name}: 0 rows")
                    continue
                for c in context_cols:
                    df_p[c] = row[c]
                df_p["team_url"] = team_url
                df_p["team_id"]  = _team_id_from_url(team_url)
                all_players.append(df_p)
                print(f"    [OK] {team_name}: {len(df_p)} players")
            except Exception as e:
                print(f"    [FAIL] {team_name}: {e}")
            time.sleep(random.uniform(*SLEEP_TEAMS))

    if not all_players:
        return pd.DataFrame()

    df = pd.concat(all_players, ignore_index=True)

    # Normalize column names: strip pg_/adv_ prefixes, prefer pg_ on conflict
    col_map = {}
    for col in df.columns:
        if col.startswith("pg_"):
            col_map[col] = col[3:]
        elif col.startswith("adv_") and col[4:] not in col_map.values():
            col_map[col] = col[4:]
    df = df.rename(columns=col_map)

    # Coerce numeric columns
    skip = {"player", "player_id", "player_href", "team", "conference",
            "conference_slug", "team_url", "team_id", "record"}
    for c in df.columns:
        if c not in skip:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _scrape_first_college_season(player_href: str, session: requests.Session):
    """Return the first college season year from a player's sports-reference page."""
    url = BASE + player_href
    try:
        html = fetch_html(url, session)
    except Exception:
        return None
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id=re.compile(r"players_per_game|per_game"))
    if table is None:
        return None
    years = []
    for cell in table.find_all(["td", "th"], attrs={"data-stat": "year_id"}):
        text = cell.get_text(strip=True)
        m = re.match(r"(\d{4})-(\d{2})", text)
        if m:
            years.append(int(text[:2]) * 100 + int(m.group(2)))
    return min(years) if years else None


def build_first_seasons_cache(force: bool = False) -> None:
    """
    Reads all raw player CSVs, finds draft-year players who only appear in
    recent data (likely transfers from unscraped conferences), fetches their
    player pages, and saves player_first_seasons.csv.

    Skipped on subsequent runs unless --force is passed, so 04_build_features
    can just do a CSV lookup with no HTTP requests.
    """
    if FIRST_SEASONS_CACHE.exists() and not force:
        print(f"  [SKIP] {FIRST_SEASONS_CACHE.name} exists — pass --force to refresh")
        return

    # Load all raw CSVs and find each player's earliest known season
    frames = []
    for year in YEARS:
        p = OUT_DIR / f"ncaaw_players_raw_{year}.csv"
        if p.exists():
            df = pd.read_csv(p, usecols=lambda c: c in
                             {"player_id", "player", "player_href", "season_year"},
                             low_memory=False)
            frames.append(df)
    if not frames:
        print("  [WARN] No raw player CSVs found — skipping first-season cache")
        return

    all_df = pd.concat(frames, ignore_index=True)
    all_df["season_year"] = pd.to_numeric(all_df["season_year"], errors="coerce")

    # One row per player: earliest season seen in our scrape + their href
    summary = (
        all_df.sort_values("season_year")
              .drop_duplicates(subset=["player_id"], keep="first")
              [["player_id", "player", "player_href", "season_year"]]
              .rename(columns={"season_year": "first_season_scraped"})
    )

    # Only check players who first appear in recent years (likely transfers)
    # and whose most recent season is the draft year
    most_recent = all_df.groupby("player_id")["season_year"].max().reset_index()
    most_recent.columns = ["player_id", "most_recent_year"]
    summary = summary.merge(most_recent, on="player_id")

    candidates = summary[
        (summary["most_recent_year"] == DRAFT_YEAR) &
        (summary["first_season_scraped"] >= DRAFT_YEAR - 1)
    ].copy()

    print(f"  Fetching player pages for {len(candidates)} possible transfer players...")
    results = []
    with requests.Session() as session:
        session.headers.update(HEADERS)
        for _, row in candidates.iterrows():
            href = str(row.get("player_href", "") or "")
            if not href.startswith("/cbb/players/"):
                continue
            time.sleep(random.uniform(3.0, 5.0))
            true_first = _scrape_first_college_season(href, session)
            scraped_first = int(row["first_season_scraped"])
            if true_first and true_first < scraped_first:
                print(f"    {row['player']}: {scraped_first} → {true_first}")
                results.append({"player_id": row["player_id"], "true_first_season": true_first})
            else:
                results.append({"player_id": row["player_id"], "true_first_season": scraped_first})

    if results:
        pd.DataFrame(results).to_csv(FIRST_SEASONS_CACHE, index=False)
        print(f"  Saved {len(results)} entries → {FIRST_SEASONS_CACHE.name}")
    else:
        print("  No transfer corrections needed.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-scrape and overwrite existing raw CSVs")
    parser.add_argument("--years", type=int, nargs="+",
                        help="Only scrape these years (e.g. --years 2025 2026)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    years_to_run = args.years if args.years else YEARS

    for year in years_to_run:
        teams_out   = OUT_DIR / f"ncaaw_teams_{year}.csv"
        players_out = OUT_DIR / f"ncaaw_players_raw_{year}.csv"

        # ---- Team standings ----
        if teams_out.exists() and not args.force:
            print(f"\n[SKIP] {teams_out.name} already exists (use --force to re-scrape)")
            teams_df = pd.read_csv(teams_out)
        else:
            print(f"\n=== Scraping teams for {year} ===")
            with requests.Session() as session:
                teams_df = scrape_teams_for_year(year, session)
            teams_df[["season_year", "conference", "team", "wins", "losses", "team_url"]].to_csv(
                teams_out, index=False
            )
            print(f"  -> {len(teams_df)} teams saved to {teams_out.name}")

        # ---- Player stats ----
        if players_out.exists() and not args.force:
            print(f"[SKIP] {players_out.name} already exists (use --force to re-scrape)")
            continue

        print(f"  Scraping players for {year} ({len(teams_df)} teams)...")
        players_df = scrape_players_for_year(teams_df)
        players_df.to_csv(players_out, index=False)
        print(f"  -> {len(players_df)} player rows saved to {players_out.name}")

    print("\n=== Building transfer first-season cache ===")
    build_first_seasons_cache(force=args.force)


if __name__ == "__main__":
    main()
