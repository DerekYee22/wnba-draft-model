"""
prepare_data.py
===============
Converts the player CSV into JSON for the React webapp.
Also embeds pre-computed readiness scores and team fit scores when the
full modeling pipeline has been run.

Usage:
    python prepare_data.py

Outputs:
    webapp/public/players.json      — player data for the frontend
"""

import csv
import json
import math
import os

# Try to load pandas for fit score merging (optional)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH          = "ncaaw_players_with_archetypes_ranked.csv"
FIT_SCORES_PATH   = os.path.join("data", "processed", "player_fit_scores.csv")
OUT_PATH          = os.path.join("webapp", "public", "players.json")

# Map the long team names (from 05_fit_scores.py) to the React team IDs
TEAM_NAME_TO_ID = {
    "Atlanta Dream":          "atlanta-dream",
    "Chicago Sky":            "chicago-sky",
    "Connecticut Sun":        "connecticut-sun",
    "Dallas Wings":           "dallas-wings",
    "Golden State Valkyries": "golden-state-valkyries",
    "Indiana Fever":          "indiana-fever",
    "Las Vegas Aces":         "las-vegas-aces",
    "Los Angeles Sparks":     "los-angeles-sparks",
    "Minnesota Lynx":         "minnesota-lynx",
    "New York Liberty":       "new-york-liberty",
    "Phoenix Mercury":        "phoenix-mercury",
    "Seattle Storm":          "seattle-storm",
    "Washington Mystics":     "washington-mystics",
}

# Columns to keep from the main CSV
KEEP_COLS = [
    "name", "pos", "team", "conference", "season_year",
    "games_played", "games_started", "mpg",
    "pts_per_g", "ast_per_g", "treb_per_g", "oreb_per_g", "dreb_per_g",
    "stl_per_g", "blk_per_g", "tov_per_g", "pf_per_g",
    "fg_pct", "fg3_pct", "ft_pct", "ts_pct", "efg_pct",
    "fg_per_g", "fga_per_g", "fg3_per_g", "fg3a_per_g",
    "per", "ws", "ws_per_40", "bpm", "obpm", "dbpm",
    "usg_pct", "oreb_pct", "dreb_pct", "treb_pct", "ast_pct", "stl_pct", "blk_pct",
    "wins", "losses", "record",
    "cluster", "archetype", "archetype_score", "rank_in_archetype",
    "pca1", "pca2",
    # Added by pipeline step 04
    "readiness_score",
]

FLOAT_COLS = {
    "mpg", "pts_per_g", "ast_per_g", "treb_per_g", "oreb_per_g", "dreb_per_g",
    "stl_per_g", "blk_per_g", "tov_per_g", "pf_per_g",
    "fg_pct", "fg3_pct", "ft_pct", "ts_pct", "efg_pct",
    "fg_per_g", "fga_per_g", "fg3_per_g", "fg3a_per_g",
    "per", "ws", "ws_per_40", "bpm", "obpm", "dbpm",
    "usg_pct", "oreb_pct", "dreb_pct", "treb_pct", "ast_pct", "stl_pct", "blk_pct",
    "archetype_score", "rank_in_archetype", "pca1", "pca2",
    "readiness_score",
}
INT_COLS = {"games_played", "games_started", "wins", "losses", "season_year", "cluster"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_float(val):
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (ValueError, TypeError):
        return None


def parse_int(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def zscore_to_100(z):
    """Convert a z-score (~-3..+3) to a 0-100 fit score for display."""
    if z is None or math.isnan(z):
        return None
    return max(0.0, min(100.0, round(50.0 + 15.0 * z, 1)))


# ---------------------------------------------------------------------------
# Load pre-computed fit scores (optional)
# ---------------------------------------------------------------------------

def load_fit_scores():
    """
    Returns dict: {player_name_lower: {team_id: 0-100 score}}
    Returns empty dict if pipeline hasn't been run or pandas is unavailable.
    """
    if not HAS_PANDAS:
        print("[INFO] pandas not available — skipping fit score embedding")
        return {}
    if not os.path.exists(FIT_SCORES_PATH):
        print(f"[INFO] {FIT_SCORES_PATH} not found — run the pipeline first for richer scores")
        return {}

    try:
        df = pd.read_csv(FIT_SCORES_PATH, low_memory=False)
    except Exception as e:
        print(f"[WARN] Could not load fit scores: {e}")
        return {}

    # Identify total-score columns (e.g. "Dallas Wings_total")
    total_cols = {
        col: col.replace("_total", "").strip()
        for col in df.columns if col.endswith("_total")
    }

    result = {}
    for _, row in df.iterrows():
        player_key = str(row.get("player", "")).lower().strip()
        if not player_key:
            continue

        scores = {}
        for col, team_name in total_cols.items():
            team_id = TEAM_NAME_TO_ID.get(team_name)
            if team_id is None:
                continue
            raw = row.get(col)
            try:
                scores[team_id] = zscore_to_100(float(raw))
            except (TypeError, ValueError):
                pass

        entry = {"fitScores": scores if scores else None}

        # Also carry readiness_score from this CSV
        rs = row.get("readiness_score")
        try:
            entry["readiness_score"] = round(float(rs), 2) if rs is not None and not math.isnan(float(rs)) else None
        except (TypeError, ValueError):
            entry["readiness_score"] = None

        result[player_key] = entry

    print(f"[INFO] Loaded pre-computed fit scores for {len(result)} players")
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run from the project root directory.")
        return

    fit_lookup = load_fit_scores()

    players = []
    skipped = 0

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        available_cols = set(reader.fieldnames or [])

        for row in reader:
            # Filter: meaningful playing time + must have archetype
            try:
                mpg      = float(row.get("mpg") or 0)
                gp       = int(float(row.get("games_played") or 0))
                rank     = float(row.get("rank_in_archetype") or 9999)
                archetype = row.get("archetype", "").strip()
            except (ValueError, TypeError):
                skipped += 1
                continue

            if mpg < 15 or gp < 12 or not archetype:
                skipped += 1
                continue

            # Only include top 300 per archetype to keep file size manageable
            if rank > 300:
                skipped += 1
                continue

            record = {}
            for col in KEEP_COLS:
                if col not in available_cols:
                    continue
                val = row.get(col, "")
                if col in FLOAT_COLS:
                    record[col] = parse_float(val)
                elif col in INT_COLS:
                    record[col] = parse_int(val)
                else:
                    record[col] = val.strip() if val else None

            # Embed pre-computed readiness_score and per-team fit scores if available.
            # Pipeline score always takes precedence over the stale value in the main CSV
            # so that running 04 → 05 → prepare_data immediately updates the webapp.
            player_key = str(row.get("name", "")).lower().strip()
            if player_key in fit_lookup:
                entry = fit_lookup[player_key]
                if entry.get("readiness_score") is not None:
                    record["readiness_score"] = entry["readiness_score"]
                record["fitScores"] = entry.get("fitScores")
            else:
                record["fitScores"] = None   # will use client-side fallback

            players.append(record)

    # Sort by rank_in_archetype
    players.sort(key=lambda p: p.get("rank_in_archetype") or 9999)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(players, f, separators=(",", ":"))

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Exported {len(players):,} players → {OUT_PATH} ({size_kb:.0f} KB)")
    print(f"Skipped {skipped:,} rows (insufficient minutes / missing archetype / rank > 300)")

    # Summary
    from collections import Counter
    counts = Counter(p["archetype"] for p in players)
    for arch, n in sorted(counts.items()):
        print(f"  {arch}: {n} players")

    has_fit = sum(1 for p in players if p.get("fitScores"))
    has_readiness = sum(1 for p in players if p.get("readiness_score") is not None)
    print(f"\nPipeline data embedded:")
    print(f"  readiness_score: {has_readiness}/{len(players)} players")
    print(f"  fitScores (pre-computed): {has_fit}/{len(players)} players")
    if has_fit == 0:
        print("  → Client-side archetype-based fit score will be used (run pipeline for full model)")


if __name__ == "__main__":
    main()
