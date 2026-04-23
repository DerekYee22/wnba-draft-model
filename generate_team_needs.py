"""
generate_team_needs.py
=======================
Converts the WNBA needs model output (wnba_top5_needs.csv) into the JSON
format used by the React webapp.

Run this after exporting wnba_top5_needs.csv from the notebook, then
run prepare_data.py to refresh the player JSON.

Usage:
    python generate_team_needs.py

Input:
    data/processed/wnba_top5_needs.csv

Output:
    webapp/public/team_needs.json
"""

import json
import math
import os

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

NEEDS_CSV   = os.path.join("data", "processed", "wnba_top5_needs.csv")
OUT_PATH    = os.path.join("webapp", "public", "team_needs.json")

# How each WNBA stat maps to player archetypes (weights 0–1).
# Archetypes match the player classifier: Floor General, Post Scorer, Combo Guard,
# 3-and-D Wing, Stretch Big, Interior Big.
STAT_TO_ARCHETYPE = {
    # Scoring efficiency deficits → Post Scorer and Combo Guard are the primary fillers
    "PTS_per_100":       {"Post Scorer": 0.45, "Combo Guard": 0.35, "Floor General": 0.20},
    "FG.":               {"Post Scorer": 0.40, "Combo Guard": 0.40, "Floor General": 0.20},
    "TS.":               {"Post Scorer": 0.50, "Combo Guard": 0.30, "3-and-D Wing": 0.20},
    # Inside scoring deficit → Post Scorer primarily, Interior Big secondary
    "percent_2FGM":      {"Post Scorer": 0.65, "Interior Big": 0.35},
    # Rim protection deficit → Interior Big primarily, Stretch Big helps secondarily
    "opp_percent_2FGM":  {"Interior Big": 0.65, "Stretch Big": 0.35},
    # Playmaking deficit → Floor General first, Combo Guard second
    "AST_per_100":       {"Floor General": 0.70, "Combo Guard": 0.30},
    # Defensive rebounding → Interior Big, Stretch Big, 3-and-D Wing
    "DRB_per_100":       {"Interior Big": 0.55, "Stretch Big": 0.30, "3-and-D Wing": 0.15},
    # Total rebounding → Interior Big, Stretch Big
    "TRB_per_100":       {"Interior Big": 0.50, "Stretch Big": 0.30, "Post Scorer": 0.20},
    # Perimeter defense deficits → 3-and-D Wing, Interior Big, Stretch Big
    "opp_FG%_per_game":  {"3-and-D Wing": 0.40, "Interior Big": 0.35, "Stretch Big": 0.25},
    "Opp_eFG":           {"3-and-D Wing": 0.45, "Interior Big": 0.35, "Stretch Big": 0.20},
    "opp_3FG%_per_game": {"3-and-D Wing": 0.65, "Floor General": 0.20, "Combo Guard": 0.15},
    # Ball disruption → Floor General, 3-and-D Wing, Interior Big
    "opp_AST_per_100":   {"Floor General": 0.40, "3-and-D Wing": 0.35, "Interior Big": 0.25},
}

STAT_DISPLAY_NAMES = {
    "PTS_per_100":        "Scoring",
    "FG.":                "Shooting Efficiency",
    "TS.":                "True Shooting",
    "Opp_eFG":            "Defensive Efficiency",
    "opp_FG%_per_game":   "Perimeter Defense",
    "opp_percent_2FGM":   "Rim Protection",
    "percent_2FGM":       "Inside Scoring",
    "AST_per_100":        "Playmaking",
    "DRB_per_100":        "Defensive Rebounding",
    "opp_3FG%_per_game":  "3-Pt Defense",
    "TRB_per_100":        "Total Rebounding",
    "opp_AST_per_100":    "Disruption",
}

ARCHETYPES = ["Floor General", "Post Scorer", "Combo Guard", "3-and-D Wing", "Stretch Big", "Interior Big"]

# Golden State Valkyries: 2025 expansion team, not in historical data
# Use moderate-to-high deficits across all stats since they're building from scratch
GSV_STATS = {
    "PTS_per_100": 0.10, "TS.": 0.09, "FG.": 0.08,
    "percent_2FGM": 0.07, "opp_percent_2FGM": 0.07,
    "AST_per_100": 0.07, "DRB_per_100": 0.06,
    "TRB_per_100": 0.06, "opp_FG%_per_game": 0.06,
    "opp_3FG%_per_game": 0.05, "opp_AST_per_100": 0.05,
}
GSV_TOP_STATS = ["Scoring", "Playmaking", "Rim Protection", "Total Rebounding"]

TEAM_NOTES = {
    "Atlanta Dream":          "Major scoring need — lacks efficient interior presence and shooting.",
    "Chicago Sky":            "Rebuilding — needs scoring and inside presence at every position.",
    "Connecticut Sun":        "Strong core; minor needs for interior defense and inside scoring.",
    "Dallas Wings":           "Defense-first rebuild — needs rim protection and perimeter stoppers.",
    "Indiana Fever":          "Have Clark for playmaking — need defenders and frontcourt support.",
    "Las Vegas Aces":         "Championship core intact — looking for perimeter defenders and depth.",
    "Los Angeles Sparks":     "Rebuilding; needs a franchise scorer and defensive anchors.",
    "Minnesota Lynx":         "Experienced core — minimal needs, looking for inside presence.",
    "New York Liberty":       "Championship contender — few gaps, targeting high-value role players.",
    "Phoenix Mercury":        "Need a disruptive defender, rebounder, and secondary playmaker.",
    "Seattle Storm":          "Transitioning — needs a scoring centerpiece and shooting efficiency.",
    "Washington Mystics":     "Full rebuild — elevated needs for scoring, rebounding, and interior presence.",
    "Golden State Valkyries": "Expansion team — building from scratch, all archetypes in demand.",
}


def compute_raw_archetype_needs(team_stats: dict) -> dict:
    """
    team_stats: {stat_name: weighted_deficit}
    Returns {archetype: raw_score} — not yet normalized to 1–10.
    """
    raw = {a: 0.0 for a in ARCHETYPES}
    for stat, deficit in team_stats.items():
        if deficit <= 0 or stat not in STAT_TO_ARCHETYPE:
            continue
        for arch, weight in STAT_TO_ARCHETYPE[stat].items():
            raw[arch] += deficit * weight
    return raw


def main():
    if not HAS_PANDAS:
        print("ERROR: pandas is required. Install it with: pip install pandas")
        return

    if not os.path.exists(NEEDS_CSV):
        print(f"ERROR: {NEEDS_CSV} not found.")
        print("Export wnba_top5_needs.csv from the WNBA_needs_model.ipynb notebook, then")
        print("place it in data/processed/wnba_top5_needs.csv")
        return

    df = pd.read_csv(NEEDS_CSV)
    print(f"Loaded {len(df)} rows from {NEEDS_CSV}")

    # 1. Compute raw archetype scores for all teams
    team_stats_map = {}
    team_top_stats = {}
    for team, grp in df.groupby("Team"):
        team_stats_map[team] = dict(zip(grp["stat"], grp["weighted_deficit"]))
        top_stats = (
            grp[grp["weighted_deficit"] > 0]
            .sort_values("weighted_deficit", ascending=False)
            .head(4)["stat"]
            .map(lambda s: STAT_DISPLAY_NAMES.get(s, s))
            .tolist()
        )
        team_top_stats[team] = top_stats

    # Add Golden State Valkyries (expansion team, not in historical data)
    team_stats_map["Golden State Valkyries"] = GSV_STATS
    team_top_stats["Golden State Valkyries"] = GSV_TOP_STATS

    all_raw = {team: compute_raw_archetype_needs(stats)
               for team, stats in team_stats_map.items()}

    # 2. Normalize in two steps:
    #    a) Within-team: scale so each team's top archetype = 10 (shows relative priorities)
    #    b) Then apply a severity ceiling: teams with few real needs cap at lower max scores
    #       This prevents a team with tiny deficits from showing all bars at 10.

    # Global max raw score (used to calibrate severity ceiling)
    all_positive = [v for scores in all_raw.values() for v in scores.values() if v > 0]
    global_max = max(all_positive) if all_positive else 1.0

    def norm_team(raw: dict) -> dict:
        team_max = max(raw.values()) if raw else 1.0
        if team_max <= 0:
            return {arch: 1.0 for arch in ARCHETYPES}
        # How severe is this team's need relative to the worst team? (0.0–1.0)
        severity = min(1.0, team_max / global_max)
        # Top archetype for a team with max severity = 10; minimal severity team = 5
        # This keeps bars visible for low-need teams while showing true scale
        ceiling = round(5.0 + 5.0 * severity, 1)
        floor = 1.0
        result = {}
        for arch in ARCHETYPES:
            v = raw[arch]
            # Scale within-team from floor to ceiling
            scaled = floor + (ceiling - floor) * (v / team_max)
            result[arch] = round(scaled, 1)
        return result

    # 3. Build output
    output = {}
    for team in sorted(all_raw.keys()):
        raw = all_raw[team]
        archetype_needs = norm_team(raw)

        output[team] = {
            "archetypeNeeds": archetype_needs,
            "topStatNeeds":   team_top_stats[team],
            "notes":          TEAM_NOTES.get(team, ""),
        }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported team needs → {OUT_PATH}")
    print(f"Teams: {len(output)}")

    # Print a summary
    for team, data in output.items():
        top = sorted(data["archetypeNeeds"].items(), key=lambda x: -x[1])
        print(f"  {team}: top need = {top[0][0]} ({top[0][1]:.1f})")


if __name__ == "__main__":
    main()
