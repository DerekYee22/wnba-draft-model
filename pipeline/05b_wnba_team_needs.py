"""
05b_wnba_team_needs.py
=======================
Scrapes WNBA team stats from basketball-reference for the specified season
and computes team stat deficits (the same computation as WNBA_needs_model.ipynb).

Run after 05_scrape_wnba_archetypes.py (or standalone).

Output:
    data/processed/wnba_top5_needs.csv   (consumed by 08_fit_scores.py and generate_team_needs.py)
"""

import time
import random
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from bs4 import BeautifulSoup, Comment

ROOT     = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
OUT_PATH = PROC_DIR / "wnba_top5_needs.csv"

# Season to use for team needs (most recent completed WNBA season)
SEASON = 2025

SLEEP_RANGE = (3.0, 6.0)
MAX_RETRIES = 4

# ---------------------------------------------------------------------------
# Stat weights and directions from WNBA_needs_model.ipynb (corr.csv).
# direction=+1 → higher is better (deficit = league_avg - team_stat)
# direction=-1 → lower is better (deficit = team_stat - league_avg)
# ---------------------------------------------------------------------------
STAT_CONFIG = {
    "PTS_per_100":       {"weight": 0.107018, "direction":  1},
    "FG.":               {"weight": 0.098936, "direction":  1},
    "opp_FG%_per_game":  {"weight": 0.097583, "direction": -1},
    "TS.":               {"weight": 0.095909, "direction":  1},
    "Opp_eFG":           {"weight": 0.084405, "direction": -1},
    "percent_2FGM":      {"weight": 0.080982, "direction":  1},
    "opp_percent_2FGM":  {"weight": 0.080982, "direction": -1},
    "AST_per_100":       {"weight": 0.077795, "direction":  1},
    "DRB_per_100":       {"weight": 0.074710, "direction":  1},
    "opp_3FG%_per_game": {"weight": 0.070988, "direction": -1},
    "TRB_per_100":       {"weight": 0.065991, "direction":  1},
    "opp_AST_per_100":   {"weight": 0.064700, "direction": -1},
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def make_session():
    try:
        import cloudscraper
        return cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "mobile": False}
        )
    except ImportError:
        s = requests.Session()
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })
        return s


def fetch_html(url: str, session) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed to fetch {url}: {e}")
            time.sleep(10.0 * attempt)
    return ""


def soups_with_comments(html: str):
    soup = BeautifulSoup(html, "html.parser")
    result = [soup]
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            result.append(BeautifulSoup(c, "html.parser"))
    return result


def find_table(soups, *table_ids):
    for tid in table_ids:
        for s in soups:
            t = s.find("table", id=tid)
            if t:
                return t
    return None


def table_to_df(table) -> pd.DataFrame:
    """Parse an HTML table into a DataFrame using data-stat attributes."""
    if table is None:
        return pd.DataFrame()
    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()
    rows = []
    for tr in tbody.find_all("tr"):
        cls = tr.get("class") or []
        if "thead" in cls or "partial_table" in cls:
            continue
        row = {}
        # Team name from th or td[data-stat="team_name"]
        th = tr.find("th")
        if th:
            a = th.find("a")
            row["team"] = a.get_text(strip=True) if a else th.get_text(strip=True)
        for td in tr.find_all("td"):
            ds = td.get("data-stat")
            if ds:
                txt = td.get_text(strip=True)
                # Also try to follow team name links
                if ds == "team_name":
                    a = td.find("a")
                    row["team"] = a.get_text(strip=True) if a else txt
                else:
                    row[ds] = txt
        if row.get("team") and row["team"] not in ("", "Team"):
            rows.append(row)
    return pd.DataFrame(rows)


def to_float(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------------------------------------------------------------------------
# Basketball-reference WNBA URL helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scrape
# ---------------------------------------------------------------------------

def scrape_team_stats(year: int, session) -> pd.DataFrame:
    """
    Scrape per-game and per-100 team + opponent stats from basketball-reference.
    All tables live on the single season page (/wnba/years/YEAR.html).
    Returns a DataFrame with one row per team and all needed features.
    """
    url = f"https://www.basketball-reference.com/wnba/years/{year}.html"
    print(f"  Fetching {year} WNBA season page...")
    html = fetch_html(url, session)
    soups = soups_with_comments(html)

    tpg   = table_to_df(find_table(soups, "per_game-team"))        # team per-game
    opg   = table_to_df(find_table(soups, "per_game-opponent"))    # opp per-game
    tp100 = table_to_df(find_table(soups, "per_poss-team"))        # team per-100
    op100 = table_to_df(find_table(soups, "per_poss-opponent"))    # opp per-100

    # Show what we have
    for name, df in [("tpg", tpg), ("tp100", tp100), ("opg", opg), ("op100", op100)]:
        print(f"    {name}: {len(df)} rows, cols: {list(df.columns)[:10]}")

    # Build the feature table
    # Normalize team names across tables
    def clean_teams(df):
        if df.empty or "team" not in df.columns:
            return df
        df = df.copy()
        df["team"] = df["team"].str.replace(r"\*", "", regex=True).str.strip()
        # Drop league average row
        df = df[~df["team"].str.lower().isin(["league average", "avg", "league avg", ""])]
        return df

    tpg   = clean_teams(tpg)
    tp100 = clean_teams(tp100)
    opg   = clean_teams(opg)
    op100 = clean_teams(op100)

    if tpg.empty:
        raise RuntimeError(f"No team per-game data scraped for {year}")

    # Build result df from per-game stats
    result = tpg[["team"]].copy()

    # FG% (team)
    if "fg_pct" in tpg.columns:
        result["FG."] = pd.to_numeric(tpg["fg_pct"].values, errors="coerce")

    # TS% — compute from pts, fga, fta
    for df_name, df in [("tpg", tpg)]:
        pts_col = next((c for c in ["pts_per_g", "pts"] if c in tpg.columns), None)
        fga_col = next((c for c in ["fga_per_g", "fga"] if c in tpg.columns), None)
        fta_col = next((c for c in ["fta_per_g", "fta"] if c in tpg.columns), None)
        if pts_col and fga_col and fta_col:
            pts = pd.to_numeric(tpg[pts_col], errors="coerce")
            fga = pd.to_numeric(tpg[fga_col], errors="coerce")
            fta = pd.to_numeric(tpg[fta_col], errors="coerce")
            result["TS."] = pts / (2 * (fga + 0.44 * fta))

    # per-100 stats
    if not tp100.empty:
        tp100 = to_float(tp100, ["pts", "ast", "trb", "drb", "pts_per_g", "ast_per_g"])
        # PTS per 100
        pts100 = next((c for c in ["pts", "pts_per_g"] if c in tp100.columns), None)
        if pts100:
            # Merge on team
            result = result.merge(
                tp100[["team", pts100]].rename(columns={pts100: "PTS_per_100"}),
                on="team", how="left"
            )
        # AST per 100
        ast100 = next((c for c in ["ast", "ast_per_g"] if c in tp100.columns), None)
        if ast100:
            result = result.merge(
                tp100[["team", ast100]].rename(columns={ast100: "AST_per_100"}),
                on="team", how="left"
            )
        # TRB per 100
        trb100 = next((c for c in ["trb", "trb_per_g"] if c in tp100.columns), None)
        if trb100:
            result = result.merge(
                tp100[["team", trb100]].rename(columns={trb100: "TRB_per_100"}),
                on="team", how="left"
            )
        # DRB per 100
        drb100 = next((c for c in ["drb", "drb_per_g"] if c in tp100.columns), None)
        if drb100:
            result = result.merge(
                tp100[["team", drb100]].rename(columns={drb100: "DRB_per_100"}),
                on="team", how="left"
            )

    # % of FGM that are 2-pointers
    fg2_col  = next((c for c in ["fg2_per_g", "fg2", "x2p"] if c in tpg.columns), None)
    fg3_col  = next((c for c in ["fg3_per_g", "fg3", "x3p"] if c in tpg.columns), None)
    fg_col   = next((c for c in ["fg_per_g", "fg"] if c in tpg.columns), None)
    if fg2_col and fg_col:
        fg2v = pd.to_numeric(tpg[fg2_col], errors="coerce")
        fgv  = pd.to_numeric(tpg[fg_col],  errors="coerce")
        result["percent_2FGM"] = fg2v / fgv
    elif fg3_col and fg_col:
        fg3v = pd.to_numeric(tpg[fg3_col], errors="coerce")
        fgv  = pd.to_numeric(tpg[fg_col],  errors="coerce")
        result["percent_2FGM"] = (fgv - fg3v) / fgv

    # Opponent stats
    def merge_opp_stat(result, opp_df, col_in, col_out, numeric=True):
        if opp_df.empty or col_in not in opp_df.columns:
            return result
        col_vals = pd.to_numeric(opp_df[col_in], errors="coerce") if numeric else opp_df[col_in]
        tmp = opp_df[["team"]].copy()
        tmp[col_out] = col_vals.values
        return result.merge(tmp, on="team", how="left")

    # Opp FG% — column is 'opp_fg_pct' in the opponent table
    opp_fgpct = next((c for c in ["opp_fg_pct", "fg_pct"] if c in opg.columns), None)
    if opp_fgpct:
        result = merge_opp_stat(result, opg, opp_fgpct, "opp_FG%_per_game")

    # Opp 3P%
    opp_fg3pct = next((c for c in ["opp_fg3_pct", "fg3_pct"] if c in opg.columns), None)
    if opp_fg3pct:
        result = merge_opp_stat(result, opg, opp_fg3pct, "opp_3FG%_per_game")

    # Opp eFG% — compute from opponent FG, 3P, FGA (opp_ prefix columns)
    ofg_c  = next((c for c in ["opp_fg", "fg"] if c in opg.columns), None)
    ofg3_c = next((c for c in ["opp_fg3", "fg3"] if c in opg.columns), None)
    ofga_c = next((c for c in ["opp_fga", "fga"] if c in opg.columns), None)
    if ofg_c and ofg3_c and ofga_c:
        fg_v  = pd.to_numeric(opg[ofg_c],  errors="coerce")
        fg3_v = pd.to_numeric(opg[ofg3_c], errors="coerce")
        fga_v = pd.to_numeric(opg[ofga_c], errors="coerce")
        tmp = opg[["team"]].copy()
        tmp["Opp_eFG"] = (fg_v + 0.5 * fg3_v) / fga_v
        result = result.merge(tmp, on="team", how="left")

    # Opp % 2-pointers
    ofg2_c = next((c for c in ["opp_fg2", "fg2"] if c in opg.columns), None)
    if ofg2_c and ofg_c:
        tmp = opg[["team"]].copy()
        tmp["opp_percent_2FGM"] = pd.to_numeric(opg[ofg2_c], errors="coerce") / pd.to_numeric(opg[ofg_c], errors="coerce")
        result = result.merge(tmp, on="team", how="left")
    elif ofg3_c and ofg_c:
        tmp = opg[["team"]].copy()
        fgv  = pd.to_numeric(opg[ofg_c],  errors="coerce")
        fg3v = pd.to_numeric(opg[ofg3_c], errors="coerce")
        tmp["opp_percent_2FGM"] = (fgv - fg3v) / fgv
        result = result.merge(tmp, on="team", how="left")

    # Opp AST per 100 — column is 'opp_ast' in opponent per-100 table
    opp_ast = next((c for c in ["opp_ast", "ast"] if c in op100.columns), None)
    if opp_ast:
        result = merge_opp_stat(result, op100, opp_ast, "opp_AST_per_100")

    result["Year"] = year
    print(f"  Built stats table: {len(result)} teams × {len(result.columns)} columns")
    print(f"  Features present: {[c for c in STAT_CONFIG if c in result.columns]}")
    return result


# ---------------------------------------------------------------------------
# Team name normalization
# ---------------------------------------------------------------------------

BREF_TO_CANONICAL = {
    # basketball-reference name → canonical name used elsewhere in the pipeline
    "Atlanta Dream":          "Atlanta Dream",
    "Chicago Sky":            "Chicago Sky",
    "Connecticut Sun":        "Connecticut Sun",
    "Dallas Wings":           "Dallas Wings",
    "Golden State Valkyries": "Golden State Valkyries",
    "Indiana Fever":          "Indiana Fever",
    "Las Vegas Aces":         "Las Vegas Aces",
    "Los Angeles Sparks":     "Los Angeles Sparks",
    "Minnesota Lynx":         "Minnesota Lynx",
    "New York Liberty":       "New York Liberty",
    "Phoenix Mercury":        "Phoenix Mercury",
    "Seattle Storm":          "Seattle Storm",
    "Washington Mystics":     "Washington Mystics",
}


# ---------------------------------------------------------------------------
# Deficit computation (mirrors WNBA_needs_model.ipynb logic)
# ---------------------------------------------------------------------------

def compute_needs(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted deficits for each team × stat.
    deficit = direction * (league_avg - team_stat) / sigma   [for direction=+1]
    deficit = direction * (team_stat - league_avg) / sigma   [= -1 case naturally flips]
    weighted_deficit = deficit * weight
    """
    records = []
    for stat, cfg in STAT_CONFIG.items():
        if stat not in team_df.columns:
            print(f"  [WARN] stat '{stat}' not found in scraped data — skipping")
            continue

        col = pd.to_numeric(team_df[stat], errors="coerce")
        mu  = col.mean()
        sigma = col.std(ddof=0)
        if sigma < 1e-8:
            continue

        direction = cfg["direction"]
        weight    = cfg["weight"]

        for _, row in team_df.iterrows():
            val = row[stat]
            if pd.isna(val):
                continue
            # z-score relative to league (positive = above average)
            z = (val - mu) / sigma
            # deficit: positive = team is weak here, negative = team is strong
            # direction=+1: weak if z<0 (below avg on good stat) → deficit = -z
            # direction=-1: weak if z>0 (above avg on bad stat) → deficit = +z
            deficit = -direction * z
            weighted_deficit = deficit * weight

            records.append({
                "Team":            row["team"],
                "stat":            stat,
                "weighted_deficit": round(weighted_deficit, 6),
                "deficit":         round(deficit, 6),
                "weight":          weight,
                "direction":       float(direction),
            })

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scraping {SEASON} WNBA team stats from basketball-reference...")
    session = make_session()

    team_df = scrape_team_stats(SEASON, session)

    # Normalize team names
    team_df["team"] = team_df["team"].map(
        lambda t: BREF_TO_CANONICAL.get(t, t)
    )

    print(f"\nTeams scraped: {sorted(team_df['team'].tolist())}")

    print("\nComputing team stat deficits...")
    needs_df = compute_needs(team_df)

    needs_df.to_csv(OUT_PATH, index=True)
    print(f"\nSaved → {OUT_PATH}")
    print(f"  {len(needs_df)} rows ({needs_df['Team'].nunique()} teams × {needs_df['stat'].nunique()} stats)")

    # Quick sanity check
    print("\nTop 3 deficits per team:")
    for team, grp in needs_df.groupby("Team"):
        top = grp[grp["weighted_deficit"] > 0].nlargest(3, "weighted_deficit")
        if not top.empty:
            items = ", ".join(f"{r['stat']}({r['weighted_deficit']:.3f})" for _, r in top.iterrows())
            print(f"  {team}: {items}")


if __name__ == "__main__":
    main()
