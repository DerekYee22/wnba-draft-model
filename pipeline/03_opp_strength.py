"""
02b_opp_strength.py
===================
Computes opponent strength metrics for each team-season from team standings CSVs.
For years where standings haven't been scraped yet (2018-2021), this script
fetches them directly from sports-reference.com before computing the metrics.

Two metrics are produced:
  conf_opp_win_pct  — average win% of all OTHER teams in the same conference.
                      Captures within-conference competition level.
  overall_opp_win_pct — average win% of ALL other D1 teams for that season.
                        Used as a fallback / cross-conference baseline.

The final per-team value written to the output is a blend:
  opp_win_pct = 0.7 * conf_opp_win_pct + 0.3 * overall_opp_win_pct

Inputs:
    data/raw/ncaaw_teams_{year}.csv   (scraped by 01_scrape_multi_year.py or
                                       fetched inline below for missing years)

Output:
    data/processed/team_opp_strength.csv
        columns: season_year, team_id, team, conference,
                 win_pct, conf_opp_win_pct, overall_opp_win_pct, opp_win_pct
"""

import re
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from bs4 import BeautifulSoup, Comment

ROOT    = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
YEARS   = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

CONF_WEIGHT    = 0.7
OVERALL_WEIGHT = 0.3

BASE    = "https://www.sports-reference.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}
CONFERENCES = [
    ("Southeastern Conference", "sec"),
    ("Big Ten Conference", "big-ten"),
    ("Atlantic Coast Conference", "acc"),
    ("Big 12 Conference", "big-12"),
    ("Big East Conference", "big-east"),
    ("American Athletic Conference", "aac"),
    ("Missouri Valley Conference", "mvc"),
    ("West Coast Conference", "wcc"),
    ("Atlantic 10 Conference", "atlantic-10"),
    ("Mountain West Conference", "mwc"),
    ("Horizon League", "horizon"),
    ("America East Conference", "america-east"),
    ("Western Athletic Conference", "wac"),
    ("Coastal Athletic Association", "coastal"),
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


def _scrape_conf_standings(conf_slug: str, year: int,
                            session: requests.Session) -> list:
    """Scrape win/loss records for all teams in a conference for a given year."""
    url = f"{BASE}/cbb/conferences/{conf_slug}/women/{year}.html"
    try:
        r = session.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    soups = [soup]
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            soups.append(BeautifulSoup(c, "html.parser"))

    table = None
    for s in soups:
        t = s.find("table", id="standings")
        if t:
            table = t
            break
    if table is None:
        for s in soups:
            for t in s.find_all("table"):
                thead = t.find("thead")
                if not thead:
                    continue
                hs = {th.get_text(strip=True) for th in thead.find_all("th")}
                if {"W", "L"}.issubset(hs) and ("School" in hs or "Team" in hs):
                    table = t
                    break
            if table:
                break
    if table is None:
        return []

    rows = []
    tbody = table.find("tbody")
    if not tbody:
        return rows
    for tr in tbody.find_all("tr"):
        cls = tr.get("class") or []
        if "thead" in cls or "spacer" in cls:
            continue
        a = tr.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        m = re.search(r"/cbb/schools/([^/]+)/women", href)
        if not m:
            continue
        team_id = m.group(1)

        def _cell(ds):
            td = tr.find(["td", "th"], attrs={"data-stat": ds})
            return td.get_text(strip=True) if td else None

        wins   = _cell("wins")   or _cell("w")
        losses = _cell("losses") or _cell("l")
        try:
            w = int(wins)
            l = int(losses)
        except (TypeError, ValueError):
            continue
        rows.append({
            "team":    a.get_text(strip=True),
            "team_id": team_id,
            "wins":    w,
            "losses":  l,
        })
    return rows


MIN_TEAMS_EXPECTED = 200   # below this → assume scrape was incomplete

def _fetch_and_cache_standings(year: int) -> pd.DataFrame:
    """Fetch standings for a year not yet in data/raw/ and cache the CSV."""
    out_path = RAW_DIR / f"ncaaw_teams_{year}.csv"

    # If a CSV already exists but is incomplete (rate-limited partial run), re-fetch.
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if len(existing) >= MIN_TEAMS_EXPECTED:
            return existing   # good enough, use cache
        print(f"  Incomplete cache ({len(existing)} teams) for {year}, re-fetching...")

    print(f"  Fetching standings for {year} from sports-reference...")
    with requests.Session() as session:
        rows = []
        for i, (conf_name, conf_slug) in enumerate(CONFERENCES):
            conf_rows = _scrape_conf_standings(conf_slug, year, session)
            for r in conf_rows:
                r["season_year"] = year
                r["conference"] = conf_name
                r["team_url"] = (
                    f"{BASE}/cbb/schools/{r['team_id']}/women/{year}.html"
                )
            rows.extend(conf_rows)
            # Longer sleep to avoid rate limiting; extra pause every 10 conferences
            sleep_time = 3.0 if (i + 1) % 10 != 0 else 8.0
            time.sleep(sleep_time)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["team_id"])
    df.to_csv(out_path, index=False)
    print(f"    -> {len(df)} teams cached to {out_path.name}")
    return df


def compute_opp_strength(year: int) -> pd.DataFrame:
    path = RAW_DIR / f"ncaaw_teams_{year}.csv"
    if not path.exists() or pd.read_csv(path).shape[0] < MIN_TEAMS_EXPECTED:
        df = _fetch_and_cache_standings(year)
        if df.empty:
            print(f"  [WARN] Could not get standings for {year}, skipping")
            return pd.DataFrame()
    else:
        df = pd.read_csv(path)

    # Extract team_id from team_url
    df["team_id"] = (
        df["team_url"]
        .str.extract(r"/cbb/schools/([^/]+)/women", expand=False)
        .fillna("")
    )

    # Win percentage (handle missing / zero-game rows)
    df["wins"]   = pd.to_numeric(df["wins"],   errors="coerce").fillna(0)
    df["losses"] = pd.to_numeric(df["losses"], errors="coerce").fillna(0)
    df["total_games"] = df["wins"] + df["losses"]
    df["win_pct"] = np.where(
        df["total_games"] > 0,
        df["wins"] / df["total_games"],
        np.nan,
    )

    # Overall season average (excluding the team itself when computing per-team)
    overall_mean = df["win_pct"].mean()

    rows = []
    for _, team_row in df.iterrows():
        conf = team_row["conference"]
        tid  = team_row["team_id"]

        # Conference peers (same conference, different team)
        conf_peers = df[
            (df["conference"] == conf) & (df["team_id"] != tid)
        ]["win_pct"]

        conf_opp_wp = conf_peers.mean() if len(conf_peers) >= 2 else overall_mean

        # Blended metric
        opp_wp = CONF_WEIGHT * conf_opp_wp + OVERALL_WEIGHT * overall_mean

        rows.append({
            "season_year":        year,
            "team_id":            tid,
            "team":               team_row["team"],
            "conference":         conf,
            "win_pct":            round(team_row["win_pct"], 4) if not np.isnan(team_row["win_pct"]) else np.nan,
            "conf_opp_win_pct":   round(conf_opp_wp, 4),
            "overall_opp_win_pct": round(overall_mean, 4),
            "opp_win_pct":        round(opp_wp, 4),
        })

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, year in enumerate(YEARS):
        print(f"  Computing opponent strength for {year}...")
        df = compute_opp_strength(year)
        if not df.empty:
            frames.append(df)
            print(f"    -> {len(df)} teams")
        # Pause between years to avoid rate limiting on sequential scrapes
        if i < len(YEARS) - 1:
            time.sleep(10.0)

    if not frames:
        print("No data produced.")
        return

    out = pd.concat(frames, ignore_index=True)
    out_path = OUT_DIR / "team_opp_strength.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} team-season rows -> {out_path.name}")

    # Quick sanity check: top/bottom conferences by opp_win_pct
    conf_summary = (
        out[out["season_year"] == max(YEARS)]
        .groupby("conference")["opp_win_pct"]
        .mean()
        .sort_values(ascending=False)
    )
    print(f"\nTop 5 conferences by opp_win_pct ({max(YEARS)}):")
    print(conf_summary.head(5).round(3).to_string())
    print(f"\nBottom 5:")
    print(conf_summary.tail(5).round(3).to_string())


if __name__ == "__main__":
    main()
