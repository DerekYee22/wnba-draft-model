"""
NCAAW Player Measurables Scraper
=================================
Scrapes height and weight from sports-reference.com player profile pages.

Input:  data/raw/ncaaw_players_raw_2025.csv  (or union of all years)
Output: data/raw/measurables_raw.csv  [player_id, player_href, height_in, weight_lbs]

Usage:
    python pipeline/02_scrape_measurables.py

Resume-safe: uses data/raw/measurables_checkpoint.txt to skip already-fetched players.
"""

import re
import time
import random
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_DIR      = Path(__file__).parent.parent / "data" / "raw"
YEARS        = [2022, 2023, 2024, 2025]
OUT_PATH     = RAW_DIR / "measurables_raw.csv"
CHECKPOINT   = RAW_DIR / "measurables_checkpoint.txt"
BASE         = "https://www.sports-reference.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sports-reference.com/",
}

MAX_RETRIES  = 3
SLEEP_RANGE  = (1.5, 3.0)
BATCH_SIZE   = 100   # save checkpoint every N players

# ---------------------------------------------------------------------------
# Height / weight parsing
# ---------------------------------------------------------------------------
HEIGHT_RE = re.compile(r"(\d+)-(\d+)")          # e.g. "6-2"
WEIGHT_RE = re.compile(r"(\d{2,3})\s*lb", re.I) # e.g. "185lb"
CM_RE     = re.compile(r"\((\d+)cm")            # e.g. "(188cm" — fallback


def parse_height(text: str):
    """Return height in inches, or None."""
    m = HEIGHT_RE.search(text)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    # fallback: centimetres
    m = CM_RE.search(text)
    if m:
        return round(int(m.group(1)) / 2.54)
    return None


def parse_weight(text: str):
    """Return weight in lbs, or None."""
    m = WEIGHT_RE.search(text)
    if m:
        return int(m.group(1))
    # fallback: kg
    kg_m = re.search(r"(\d{2,3})\s*kg", text, re.I)
    if kg_m:
        return round(int(kg_m.group(1)) * 2.205)
    return None


def scrape_measurables_for_player(player_id: str, session: requests.Session):
    """
    Fetch a player's profile page and parse height / weight.
    Returns dict with keys: player_id, height_in, weight_lbs.
    """
    url = f"{BASE}/cbb/players/{player_id}.html"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                return {"player_id": player_id, "height_in": None, "weight_lbs": None,
                        "error": str(e)}
            time.sleep(1.5 * attempt)

    soup = BeautifulSoup(r.text, "html.parser")

    # Sports-Reference player bio is in div#info > p tags
    info_div = soup.find("div", id="info")
    height_in  = None
    weight_lbs = None

    if info_div:
        for p in info_div.find_all("p"):
            text = p.get_text(" ", strip=True)
            if height_in is None:
                height_in = parse_height(text)
            if weight_lbs is None:
                weight_lbs = parse_weight(text)
            if height_in and weight_lbs:
                break

    return {"player_id": player_id, "height_in": height_in, "weight_lbs": weight_lbs}

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        return set(CHECKPOINT.read_text().splitlines())
    return set()


def save_checkpoint(done: set):
    CHECKPOINT.write_text("\n".join(sorted(done)))


def load_existing_results() -> list:
    if OUT_PATH.exists():
        return pd.read_csv(OUT_PATH).to_dict("records")
    return []

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_player_ids() -> pd.DataFrame:
    """Union player_id + player_href across all available year CSVs."""
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"ncaaw_players_raw_{year}.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=lambda c: c in ("player_id", "player_href", "player"))
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No raw player CSVs found in data/raw/. Run 01_scrape_multi_year.py first."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])
    print(f"Found {len(combined)} unique player IDs across {len(frames)} year file(s)")
    return combined


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    players    = collect_player_ids()
    done       = load_checkpoint()
    results    = load_existing_results()
    results_by_id = {r["player_id"]: r for r in results}

    to_fetch = [pid for pid in players["player_id"] if pid not in done]
    print(f"Already done: {len(done)} | Remaining: {len(to_fetch)}")

    with requests.Session() as session:
        for i, player_id in enumerate(to_fetch, 1):
            rec = scrape_measurables_for_player(player_id, session)
            results_by_id[player_id] = rec
            done.add(player_id)

            print(f"  [{i}/{len(to_fetch)}] {player_id} "
                  f"height={rec.get('height_in')} weight={rec.get('weight_lbs')}")

            # Save checkpoint every BATCH_SIZE players
            if i % BATCH_SIZE == 0:
                pd.DataFrame(list(results_by_id.values())).to_csv(OUT_PATH, index=False)
                save_checkpoint(done)
                print(f"  [checkpoint saved at {i}]")

            time.sleep(random.uniform(*SLEEP_RANGE))

    # Final save
    pd.DataFrame(list(results_by_id.values())).to_csv(OUT_PATH, index=False)
    save_checkpoint(done)

    df = pd.read_csv(OUT_PATH)
    filled = df[["height_in", "weight_lbs"]].notna().sum()
    print(f"\nDone. {len(df)} players saved to {OUT_PATH.name}")
    print(f"  height_in filled:  {filled['height_in']}/{len(df)}")
    print(f"  weight_lbs filled: {filled['weight_lbs']}/{len(df)}")


if __name__ == "__main__":
    main()
