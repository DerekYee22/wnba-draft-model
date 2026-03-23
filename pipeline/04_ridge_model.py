"""
Ridge Regression — WNBA Readiness Score
=========================================
Trains a Ridge regression model to produce a competition-adjusted "WNBA readiness"
score for each NCAA player.

Why Ridge? Basketball stats (PPG, USG%, FGA) are highly collinear. Ridge shrinks
correlated coefficients toward each other rather than letting any single stat
dominate, producing a more stable composite ranking.

Target construction:
    composite_quality = f(ws_per_40, bpm, per, ts_pct) × competition_multiplier

This is not a true "WNBA success" prediction (we lack WNBA outcome labels for
current prospects). It is a competition-adjusted quality score derived from
advanced NCAA metrics, used as the Ridge model's training target. The Ridge
model then learns stable weights for the raw counting stats — the readiness_score
output is the result of applying those weights to new players.

Inputs:
    data/processed/ncaaw_players_features.csv

Outputs:
    data/processed/ncaaw_players_features.csv   (adds readiness_score column)
    data/models/ridge_model.pkl
    data/models/scaler.pkl
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "data" / "models"
IN_PATH   = PROC_DIR / "ncaaw_players_features.csv"
OUT_PATH  = PROC_DIR / "ncaaw_players_features.csv"   # overwrite in place

# Minimum games to include a player in the training set
MIN_GAMES_TRAIN = 15

# Ridge alpha grid
ALPHAS = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]

# Feature columns for Ridge
FEATURES = [
    # Per-game production
    "pts_per_g", "ast_per_g", "treb_per_g", "stl_per_g", "blk_per_g", "tov_per_g",
    "fg3a_per_g", "fg3_pct", "ft_pct", "fta_per_g", "oreb_per_g",
    # Efficiency
    "ts_pct", "efg_pct", "usg_pct",
    # Rate / role
    "ast_pct", "treb_pct", "blk_pct", "stl_pct", "tov_pct",
    # Advanced
    "per", "obpm", "dbpm",
    # Context
    "adj_opp_win_pct", "mpg_latest", "is_multi_year",
    # Derived
    "ast_tov_ratio", "three_rate", "inside_score", "def_composite",
    # Trend (0 for single-year players)
    "trend_bpm", "trend_pts_per_g",
    # Measurables (optional — imputed with archetype median if missing)
    "height_in", "weight_lbs",
]

# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_composite_target(df: pd.DataFrame) -> pd.Series:
    """
    Build a competition-adjusted composite quality score as the Ridge target.

    Components:
        ws_per_40  (0.30) — efficiency per 40 minutes, context-adjusted
        bpm        (0.30) — box plus/minus, context-independent
        per        (0.20) — player efficiency rating
        ts_pct     (0.20) — true shooting %, measures scoring quality

    All components are z-scored before combining so no single metric dominates.
    The competition multiplier rewards players in tougher conferences.
    """
    # Z-score each component robustly (clip extreme outliers first)
    def zscore(s, lo, hi):
        s = s.clip(lo, hi)
        mu, sigma = s.mean(), s.std()
        if sigma < 1e-8:
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sigma

    ws40   = zscore(df["ws_per_40"].fillna(0.0), -0.3, 0.6)
    bpm    = zscore(df["bpm"].fillna(0.0),       -15,  20)
    per    = zscore(df["per"].fillna(15.0),        5,  40)
    ts     = zscore(df["ts_pct"].fillna(0.5),      0.3, 0.75)

    raw = 0.30 * ws40 + 0.30 * bpm + 0.20 * per + 0.20 * ts

    # Competition multiplier: scale adj_opp_win_pct to [0.75, 1.25]
    if "adj_opp_win_pct" in df.columns:
        opp = pd.to_numeric(df["adj_opp_win_pct"], errors="coerce").fillna(
            df["adj_opp_win_pct"].median() if "adj_opp_win_pct" in df.columns else 0.5
        )
        opp_z = zscore(opp, opp.quantile(0.05), opp.quantile(0.95))
        # Clip multiplier so weak-conference players aren't completely zeroed out
        comp_mult = (1.0 + 0.25 * opp_z).clip(0.75, 1.25)
        raw = raw * comp_mult

    return raw.rename("composite_quality")

# ---------------------------------------------------------------------------
# Train Ridge
# ---------------------------------------------------------------------------

def train_ridge(df: pd.DataFrame):
    # Only train on players with enough games
    games_col = df.get("career_games", df.get("games_played"))
    if games_col is not None:
        train_mask = pd.to_numeric(games_col, errors="coerce").fillna(0) >= MIN_GAMES_TRAIN
    else:
        train_mask = pd.Series(True, index=df.index)

    target = build_composite_target(df)

    # Also require that the target components exist (drop rows where all are NaN)
    target_defined = df[["ws_per_40", "bpm", "per", "ts_pct"]].notna().any(axis=1)
    train_mask = train_mask & target_defined

    X_cols_present = [c for c in FEATURES if c in df.columns]
    X_train_raw = df.loc[train_mask, X_cols_present]

    # Drop columns that are entirely NaN (e.g. height/weight before measurables are scraped)
    all_nan_cols = [c for c in X_cols_present if X_train_raw[c].isna().all()]
    if all_nan_cols:
        print(f"  [INFO] Dropping all-NaN features: {all_nan_cols}")
        X_cols_present = [c for c in X_cols_present if c not in all_nan_cols]

    X_train = df.loc[train_mask, X_cols_present]
    y_train = target.loc[train_mask]

    print(f"  Training on {train_mask.sum()} players, {len(X_cols_present)} features")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("ridge",   RidgeCV(alphas=ALPHAS, cv=5, scoring="r2")),
    ])
    pipeline.fit(X_train, y_train)

    best_alpha = pipeline.named_steps["ridge"].alpha_
    cv_r2      = pipeline.named_steps["ridge"].best_score_
    print(f"  Best alpha: {best_alpha}  |  CV R²: {cv_r2:.3f}")
    print("  Note: High CV R² is expected (target is derived from input features). "
          "Validate with rank stability (Kendall's τ across folds) instead.")

    # Score ALL players (including those below the training threshold)
    X_all = df[X_cols_present]
    scores = pipeline.predict(X_all)

    # Normalise to 0–100 scale for readability in the app
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores_norm = 100 * (scores - s_min) / (s_max - s_min)
    else:
        scores_norm = np.full_like(scores, 50.0)

    return pipeline, scores_norm, X_cols_present


# ---------------------------------------------------------------------------
# Print coefficient summary
# ---------------------------------------------------------------------------

def print_coef_summary(pipeline, feature_names: list):
    coefs = pipeline.named_steps["ridge"].coef_
    summary = (
        pd.Series(coefs, index=feature_names)
          .abs()
          .sort_values(ascending=False)
          .head(15)
    )
    print("\n  Top 15 Ridge features (by |coefficient|):")
    for feat, val in summary.items():
        sign = "+" if pipeline.named_steps["ridge"].coef_[feature_names.index(feat)] > 0 else "-"
        print(f"    {sign}{val:.4f}  {feat}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"{IN_PATH} not found. Run pipeline/03_build_features.py first."
        )

    print(f"Loading {IN_PATH.name}...")
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"  {len(df)} players loaded")

    print("Building Ridge model...")
    pipeline, readiness_scores, feature_names = train_ridge(df)
    print_coef_summary(pipeline, feature_names)

    # Attach scores to dataframe
    df["readiness_score"] = readiness_scores.round(2)

    # Save updated dataframe
    df.to_csv(OUT_PATH, index=False)
    print(f"\n  readiness_score added to {OUT_PATH.name}")

    # Save model artifacts
    joblib.dump(pipeline, MODEL_DIR / "ridge_model.pkl")
    print(f"  Model saved to data/models/ridge_model.pkl")

    # Quick sanity: show top 20 by readiness_score
    cols_show = ["player", "archetype", "pts_per_g", "bpm", "ts_pct", "adj_opp_win_pct", "readiness_score"]
    cols_show = [c for c in cols_show if c in df.columns]
    top20 = df.nlargest(20, "readiness_score")[cols_show]
    print("\n  Top 20 players by readiness_score:")
    print(top20.to_string(index=False))


if __name__ == "__main__":
    main()
