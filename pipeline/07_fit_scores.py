"""
Player–Team Fit Score Calculator
==================================
Computes a fit score for every (player, WNBA team) pair by mapping WNBA team
stat deficits onto NCAA player strengths.

How it works:
    1. Load WNBA team needs (data/processed/wnba_top5_needs.csv or hardcoded fallback)
    2. Z-score all relevant NCAA player stats across the full player pool
    3. For each team need stat, map to one or more NCAA proxy stats
    4. fit_score(player, team) = Σ over needs: weighted_deficit × Σ (ncaa_z × weight × confidence)
    5. total_score(player, team) = READINESS_WEIGHT × zscore(readiness_score)
                                 + (1 - READINESS_WEIGHT) × zscore(fit_score)

Inputs:
    data/processed/ncaaw_players_features.csv  (with readiness_score from step 04)
    data/processed/wnba_top5_needs.csv         (exported from WNBA_needs_model.ipynb)

Output:
    data/processed/player_fit_scores.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
PROC_DIR    = ROOT / "data" / "processed"
PLAYERS_IN  = PROC_DIR / "ncaaw_players_features.csv"
NEEDS_IN    = PROC_DIR / "wnba_top5_needs.csv"
OUT_PATH    = PROC_DIR / "player_fit_scores.csv"

# Weight for the global readiness score vs. team-specific fit
READINESS_WEIGHT = 0.55

# ---------------------------------------------------------------------------
# WNBA stat → NCAA player stat mapping
# Each entry: wnba_stat -> list of (ncaa_stat, weight, confidence)
#   weight:     relative importance among the NCAA proxies (should sum to 1.0)
#   confidence: how direct the mapping is (1.0 = clean, 0.7 = proxy)
# ---------------------------------------------------------------------------
WNBA_TO_NCAA_MAP = {
    "PTS_per_100": [
        ("pts_per_g",  0.50, 1.0),
        ("usg_pct",    0.30, 1.0),
        ("obpm",       0.20, 1.0),
    ],
    "FG.": [
        ("fg_pct",     0.40, 1.0),
        ("efg_pct",    0.40, 1.0),
        ("ts_pct",     0.20, 1.0),
    ],
    # Opponent FG% = team defense → best NCAA proxies are rim protection + on-ball defense
    "opp_FG%_per_game": [
        ("blk_per_g",  0.40, 0.7),
        ("blk_pct",    0.30, 0.7),
        ("dbpm",       0.30, 0.7),
    ],
    "TS.": [
        ("ts_pct",     0.60, 1.0),
        ("efg_pct",    0.40, 1.0),
    ],
    # Opponent eFG% = perimeter + interior defense
    "Opp_eFG": [
        ("blk_per_g",  0.35, 0.7),
        ("stl_per_g",  0.35, 0.7),
        ("dbpm",       0.30, 0.7),
    ],
    # % of shots that are 2s = inside scoring / post presence
    "percent_2FGM": [
        ("fg2_per_g",  0.50, 1.0),
        ("oreb_per_g", 0.30, 0.8),
        ("fg2_pct",    0.20, 1.0),
    ],
    # Opponent % of 2s = rim protection
    "opp_percent_2FGM": [
        ("blk_per_g",  0.50, 0.7),
        ("treb_pct",   0.30, 0.8),
        ("dws",        0.20, 0.7),
    ],
    "AST_per_100": [
        ("ast_per_g",  0.55, 1.0),
        ("ast_pct",    0.45, 1.0),
    ],
    "DRB_per_100": [
        ("dreb_per_g", 0.50, 1.0),
        ("dreb_pct",   0.25, 1.0),
        ("treb_pct",   0.25, 1.0),
    ],
    # Opponent 3FG% = perimeter defense
    "opp_3FG%_per_game": [
        ("stl_per_g",  0.45, 0.7),
        ("stl_pct",    0.35, 0.7),
        ("dbpm",       0.20, 0.7),
    ],
    "TRB_per_100": [
        ("treb_per_g", 0.55, 1.0),
        ("treb_pct",   0.45, 1.0),
    ],
    # Opponent AST = disrupting ball movement
    "opp_AST_per_100": [
        ("stl_per_g",  0.50, 0.7),
        ("blk_per_g",  0.25, 0.7),
        ("dbpm",       0.25, 0.7),
    ],
}

# ---------------------------------------------------------------------------
# Hardcoded WNBA team needs (from WNBA_needs_model.ipynb, 2024 season)
# Use this when wnba_top5_needs.csv is not available.
# Format: {team: {stat: weighted_deficit}}
# ---------------------------------------------------------------------------
HARDCODED_NEEDS = {
    "Atlanta Dream": {
        "FG.": 0.2146, "TS.": 0.1536, "percent_2FGM": 0.1435,
        "opp_percent_2FGM": 0.1435, "PTS_per_100": 0.1370,
    },
    "Chicago Sky": {
        "TS.": 0.1799, "percent_2FGM": 0.1401, "opp_percent_2FGM": 0.1401,
        "PTS_per_100": 0.1339, "FG.": 0.1144,
    },
    "Connecticut Sun": {
        "TS.": 0.0135, "DRB_per_100": 0.0111, "percent_2FGM": 0.0095,
        "opp_percent_2FGM": 0.0095, "AST_per_100": -0.0046,
    },
    "Dallas Wings": {
        "opp_FG%_per_game": 0.2141, "Opp_eFG": 0.1756, "DRB_per_100": 0.1396,
        "opp_AST_per_100": 0.1131, "opp_3FG%_per_game": 0.1029,
    },
    "Indiana Fever": {
        "opp_3FG%_per_game": 0.0880, "Opp_eFG": 0.0531, "opp_AST_per_100": 0.0353,
        "AST_per_100": 0.0229, "opp_FG%_per_game": 0.0182,
    },
    "Las Vegas Aces": {
        "opp_3FG%_per_game": 0.0471, "TRB_per_100": 0.0290, "AST_per_100": 0.0183,
        "Opp_eFG": -0.0271, "opp_FG%_per_game": -0.0278,
    },
    "Los Angeles Sparks": {
        "opp_FG%_per_game": 0.1507, "PTS_per_100": 0.1492, "Opp_eFG": 0.1165,
        "FG.": 0.1073, "opp_3FG%_per_game": 0.0843,
    },
    "Minnesota Lynx": {
        "percent_2FGM": 0.0060, "opp_percent_2FGM": 0.0060, "TRB_per_100": -0.0049,
        "PTS_per_100": -0.0344, "DRB_per_100": -0.0465,
    },
    "New York Liberty": {
        "opp_AST_per_100": -0.0468, "opp_3FG%_per_game": -0.0496, "FG.": -0.0715,
        "opp_FG%_per_game": -0.0739, "Opp_eFG": -0.0778,
    },
    "Phoenix Mercury": {
        "opp_AST_per_100": 0.0872, "TRB_per_100": 0.0751, "AST_per_100": 0.0321,
        "DRB_per_100": 0.0244, "opp_3FG%_per_game": 0.0174,
    },
    "Seattle Storm": {
        "TS.": 0.0485, "DRB_per_100": 0.0288, "FG.": 0.0215,
        "AST_per_100": 0.0092, "TRB_per_100": 0.0044,
    },
    "Washington Mystics": {
        "PTS_per_100": 0.1155, "TRB_per_100": 0.1089, "DRB_per_100": 0.0908,
        "percent_2FGM": 0.0438, "opp_percent_2FGM": 0.0438,
    },
    # Golden State Valkyries: new 2025 expansion team — neutral needs (league average)
    "Golden State Valkyries": {
        "PTS_per_100": 0.05, "TS.": 0.05, "DRB_per_100": 0.04,
        "AST_per_100": 0.04, "opp_FG%_per_game": 0.04,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zscore_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return a DataFrame of z-scores for the given columns."""
    z = pd.DataFrame(index=df.index)
    for col in cols:
        if col not in df.columns:
            z[col] = 0.0
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        mu, sigma = s.mean(), s.std()
        if sigma < 1e-8:
            z[col] = 0.0
        else:
            z[col] = ((s - mu) / sigma).fillna(0.0)
    return z


def load_team_needs() -> dict:
    """
    Load WNBA team needs.
    Returns {team: {stat: weighted_deficit}} dict.
    """
    if NEEDS_IN.exists():
        df = pd.read_csv(NEEDS_IN)
        needs = {}
        for team, grp in df.groupby("Team"):
            needs[team] = dict(zip(grp["stat"], grp["weighted_deficit"]))
        print(f"  Team needs loaded from {NEEDS_IN.name} ({len(needs)} teams)")
        return needs
    else:
        print(f"  [INFO] {NEEDS_IN.name} not found — using hardcoded 2024 team needs")
        return HARDCODED_NEEDS


def compute_fit_matrix(z_df: pd.DataFrame, team_needs: dict) -> pd.DataFrame:
    """
    Vectorized fit score computation.
    Returns DataFrame shape (n_players, n_teams) with fit scores.
    """
    # All NCAA stat columns we'll need
    all_ncaa_stats = {stat for mappings in WNBA_TO_NCAA_MAP.values()
                      for stat, _, _ in mappings}

    # Pre-build a (n_players, n_ncaa_stats) matrix
    ncaa_matrix = pd.DataFrame({s: z_df[s] if s in z_df.columns
                                 else pd.Series(0.0, index=z_df.index)
                                 for s in all_ncaa_stats})

    fit_scores = {}
    for team, needs_dict in team_needs.items():
        team_fit = pd.Series(0.0, index=z_df.index)

        for wnba_stat, deficit in needs_dict.items():
            if deficit <= 0:  # only address real weaknesses
                continue
            if wnba_stat not in WNBA_TO_NCAA_MAP:
                continue

            # Weighted sum of NCAA proxy z-scores
            ncaa_composite = pd.Series(0.0, index=z_df.index)
            total_wc = 0.0
            for ncaa_stat, weight, confidence in WNBA_TO_NCAA_MAP[wnba_stat]:
                if ncaa_stat in ncaa_matrix.columns:
                    ncaa_composite += weight * confidence * ncaa_matrix[ncaa_stat]
                    total_wc += weight * confidence

            if total_wc > 0:
                ncaa_composite /= total_wc

            team_fit += deficit * ncaa_composite

        fit_scores[team] = team_fit

    return pd.DataFrame(fit_scores)


def build_reliability(players_df: pd.DataFrame) -> pd.Series:
    """
    Volume-based reliability factor in [0.15, 1.0].

    Advanced stats like BPM, TS%, and blk_pct are very noisy for players
    with few minutes — a player who shoots 5-for-6 in garbage time looks
    like an elite shooter. This factor dampens the fit score for low-minute
    players so volume and genuine impact are required to rank highly.

    20 MPG = full credit. 10 MPG = 50% credit. Floor at 15% so even
    low-minute players can still appear in rankings (just ranked lower).
    """
    # Use mpg_latest (most recent season MPG) as the primary signal
    mpg = pd.to_numeric(
        players_df.get("mpg_latest", players_df.get("mpg", pd.Series(20, index=players_df.index))),
        errors="coerce",
    ).fillna(0)

    # 20 MPG = 1.0, 10 MPG = 0.5, 5 MPG = 0.25
    mpg_factor = (mpg / 20).clip(0, 1)

    return mpg_factor.clip(0.15, 1.0)


def combine_scores(
    players_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    readiness_z: pd.Series,
    readiness_weight: float,
) -> pd.DataFrame:
    """
    Z-score the fit scores per team and combine with readiness_score.
    The fit component is multiplied by a reliability factor before z-scoring
    so that low-minute players (with noisy advanced stats) rank lower.
    """
    result = players_df.copy()
    reliability = build_reliability(players_df)

    for team in fit_df.columns:
        # Dampen the raw fit score by playing-time reliability before z-scoring.
        # This prevents a 10 MPG player with a great BPM from floating to the top.
        raw_fit_damped = fit_df[team] * reliability

        mu, sigma = raw_fit_damped.mean(), raw_fit_damped.std()
        fit_z = ((raw_fit_damped - mu) / sigma).fillna(0.0) if sigma > 1e-8 else pd.Series(0.0, index=raw_fit_damped.index)

        result[f"{team}_fit"]   = raw_fit_damped.round(4)
        result[f"{team}_total"] = (
            readiness_weight * readiness_z
            + (1 - readiness_weight) * fit_z
        ).round(4)

    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    if not PLAYERS_IN.exists():
        raise FileNotFoundError(
            f"{PLAYERS_IN} not found. Run pipeline/06_xgboost_model.py first."
        )

    print(f"Loading players from {PLAYERS_IN.name}...")
    df = pd.read_csv(PLAYERS_IN, low_memory=False)
    print(f"  {len(df)} players")

    # Collect all NCAA proxy stats we need
    all_ncaa_cols = list({s for mappings in WNBA_TO_NCAA_MAP.values()
                          for s, _, _ in mappings})

    print("Computing player z-scores...")
    z_df = zscore_df(df, all_ncaa_cols)

    # Z-score the readiness score
    if "readiness_score" in df.columns:
        rs = pd.to_numeric(df["readiness_score"], errors="coerce").fillna(0.0)
        mu, sigma = rs.mean(), rs.std()
        readiness_z = ((rs - mu) / sigma).fillna(0.0) if sigma > 1e-8 else pd.Series(0.0)
    else:
        print("  [WARN] readiness_score not found — run 04_xgboost_model.py first")
        readiness_z = pd.Series(0.0, index=df.index)

    print("Loading WNBA team needs...")
    team_needs = load_team_needs()
    print(f"  Teams: {list(team_needs.keys())}")

    print("Computing fit score matrix...")
    fit_df = compute_fit_matrix(z_df, team_needs)

    print("Combining readiness + fit scores...")
    keep_cols = [c for c in [
        "player", "player_id", "pos", "team", "conference", "archetype", "cluster",
        "pts_per_g", "ast_per_g", "treb_per_g", "blk_per_g", "stl_per_g",
        "ts_pct", "bpm", "usg_pct", "adj_opp_win_pct",
        "readiness_score", "n_seasons", "height_in", "weight_lbs",
        "trend_bpm", "trend_pts_per_g",
    ] if c in df.columns]

    output_df = combine_scores(df[keep_cols].copy(), fit_df, readiness_z, READINESS_WEIGHT)
    output_df.to_csv(OUT_PATH, index=False)

    print(f"\nFit scores saved: {OUT_PATH.name}")
    print(f"  {len(output_df)} players × {len(team_needs)} teams")

    # Quick sanity: top 5 fits for Dallas Wings (biggest defensive need)
    if "Dallas Wings_total" in output_df.columns:
        top5 = output_df.nlargest(5, "Dallas Wings_total")[
            ["player", "archetype", "bpm", "blk_per_g", "Dallas Wings_total"]
        ]
        print("\n  Top 5 fits for Dallas Wings (biggest defensive need):")
        print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
