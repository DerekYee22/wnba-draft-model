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

# How each WNBA stat maps to player archetypes (weights 0–1)
STAT_TO_ARCHETYPE = {
    "PTS_per_100":        {"Primary Creator": 1.0, "Balanced Contributor": 0.5},
    "FG.":                {"Primary Creator": 0.8, "Balanced Contributor": 0.5},
    "TS.":                {"Primary Creator": 0.7, "Balanced Contributor": 0.7},
    "Opp_eFG":            {"Interior Defender": 0.8, "Support Player": 0.4},
    "opp_FG%_per_game":   {"Interior Defender": 1.0},
    "opp_percent_2FGM":   {"Interior Defender": 1.0},
    "percent_2FGM":       {"Interior Defender": 0.6, "Balanced Contributor": 0.5},
    "AST_per_100":        {"Support Player": 1.0, "Balanced Contributor": 0.5},
    "DRB_per_100":        {"Interior Defender": 0.8, "Support Player": 0.3},
    "opp_3FG%_per_game":  {"Support Player": 0.6, "Balanced Contributor": 0.5},
    "TRB_per_100":        {"Interior Defender": 0.9, "Balanced Contributor": 0.3},
    "opp_AST_per_100":    {"Interior Defender": 0.5, "Support Player": 0.7},
}

STAT_DISPLAY_NAMES = {
    "PTS_per_100":        "Scoring",
    "FG.":                "Shooting Efficiency",
    "TS.":                "True Shooting",
    "Opp_eFG":            "Interior Defense",
    "opp_FG%_per_game":   "Perimeter Defense",
    "opp_percent_2FGM":   "Rim Protection",
    "percent_2FGM":       "2-Pt Creation",
    "AST_per_100":        "Playmaking",
    "DRB_per_100":        "Defensive Rebounding",
    "opp_3FG%_per_game":  "3-Pt Defense",
    "TRB_per_100":        "Total Rebounding",
    "opp_AST_per_100":    "Disruption",
}

ARCHETYPES = ["Primary Creator", "Balanced Contributor", "Interior Defender", "Support Player"]


def compute_archetype_needs(team_stats: dict) -> dict:
    """
    team_stats: {stat_name: weighted_deficit}
    Returns {archetype: score_0_to_10}
    """
    raw = {a: 0.0 for a in ARCHETYPES}
    for stat, deficit in team_stats.items():
        if deficit <= 0 or stat not in STAT_TO_ARCHETYPE:
            continue
        for arch, weight in STAT_TO_ARCHETYPE[stat].items():
            raw[arch] += deficit * weight

    max_val = max(raw.values()) if any(v > 0 for v in raw.values()) else 1.0
    if max_val == 0:
        return {a: 5 for a in ARCHETYPES}

    return {
        arch: round(min(10.0, max(1.0, (val / max_val) * 10)), 1)
        for arch, val in raw.items()
    }


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

    output = {}

    for team, grp in df.groupby("Team"):
        # Build {stat: weighted_deficit} for this team
        team_stats = dict(zip(grp["stat"], grp["weighted_deficit"]))

        # Compute archetype need scores (0–10)
        arch_needs = compute_archetype_needs(team_stats)

        # Top stat labels for display (positive deficits only)
        top_stats = (
            grp[grp["weighted_deficit"] > 0]
            .sort_values("weighted_deficit", ascending=False)
            .head(4)["stat"]
            .map(lambda s: STAT_DISPLAY_NAMES.get(s, s))
            .tolist()
        )

        output[team] = {
            "archetypeNeeds": arch_needs,
            "topStatNeeds":   top_stats,
            "rawDeficits":    {
                row["stat"]: round(float(row["weighted_deficit"]), 4)
                for _, row in grp.iterrows()
                if float(row["weighted_deficit"]) > 0
            },
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
