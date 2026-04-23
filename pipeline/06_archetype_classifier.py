"""
06_archetype_classifier.py
===========================
Trains a supervised archetype classifier using WNBA player archetypes (from
step 05) as ground-truth labels and pre-draft college stats + height as
features. The trained classifier is then applied to all current college
prospects to assign their predicted WNBA archetype.

This replaces the hand-tuned GMM signature approach (old step 05). The
key improvement is that archetype assignments are now grounded in what
role players actually played in the WNBA, not in manually specified
statistical profiles.

Why logistic regression?
  - Coefficients are directly interpretable: which college stats push a player
    toward each archetype.
  - Regularised (L2/C=1.0) to handle the small training set (~150–200 matched
    players across 6 classes).
  - Calibrated probabilities useful for soft/uncertain assignments.
  - Fast enough that we can retrain every pipeline run without caching the model.

Why height?
  Height is the strongest single predictor of WNBA role. Without it, players
  like JuJu Watkins (guard who blocks shots and scores in the paint) get
  misclassified as interior bigs. Height disambiguates statistical profiles
  that look similar across positions.

Inputs:
    data/processed/wnba_player_archetypes.csv  (from 05_scrape_wnba_archetypes.py)
    data/raw/wnba_draft_history.csv            (from 06→07 xgboost, if available)
    data/processed/ncaaw_players_features.csv  (from 04_build_features.py)

Outputs:
    data/processed/ncaaw_players_features.csv          (archetype columns updated)
    ncaaw_players_with_archetypes_ranked.csv           (project root)
    data/models/archetype_classifier.pkl               (saved classifier)
"""

import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
PROC_DIR  = ROOT / "data" / "processed"
RAW_DIR   = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "data" / "models"

WNBA_ARCH_CSV = PROC_DIR / "wnba_player_archetypes.csv"
DRAFT_CSV     = RAW_DIR  / "wnba_draft_history.csv"
FEAT_CSV      = PROC_DIR / "ncaaw_players_features.csv"
OUT_ROOT      = ROOT     / "ncaaw_players_with_archetypes_ranked.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Minimum weighted-average MPG for a college player to be classified.
MIN_MPG_CLUSTER = 20

# College features used as classifier inputs.
# Mirrors the old GMM cluster features plus height_in.
CLASSIFIER_FEATURES = [
    "three_rate",
    "inside_score",
    "fta_per_g",
    "fg3_pct",
    "treb_pct",
    "oreb_pct",
    "blk_per_g",
    "usg_pct",
    "ast_pct",
    "ast_tov_ratio",  # floor generals have high ratio; combo guards are more turnover-prone
    "pts_per_g",      # scoring volume separates combo guards from pure playmakers
    "tov_per_g",
    "pos_encoded",
    "height_in",   # physical measurement — strongest position separator
]

# pct columns that may be stored as 0–100 in the college CSV
_PCT_COLS = {"usg_pct", "ts_pct", "fg3_pct", "ast_pct", "treb_pct", "oreb_pct"}

_POS_ENCODING = {
    "G": 0.0, "G-F": 0.25, "F-G": 0.25,
    "F": 0.5, "F-C": 0.75, "C-F": 0.75, "C": 1.0,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_pos(series: pd.Series) -> pd.Series:
    def _parse(val):
        if not isinstance(val, str):
            return np.nan
        m = re.search(r'\b([GFC](?:-[GFC])?)\b', val)
        return _POS_ENCODING.get(m.group(1), np.nan) if m else np.nan
    enc = series.apply(_parse)
    n_missing = enc.isna().sum()
    if n_missing:
        print(f"  [pos_encoded] imputing {n_missing} missing values → 0.4")
    return enc.fillna(0.4)


def _clean_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z\s]", "", name)
    return re.sub(r"\s+", " ", name)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _build_feature_matrix(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cols:
        if c in _PCT_COLS:
            med = np.nanmedian(X[c].values)
            if med > 1.5:
                X[c] = X[c] / 100.0
    return X

# ---------------------------------------------------------------------------
# Step 1: Build training set by matching WNBA players to college stats
# ---------------------------------------------------------------------------

def build_training_set(wnba_df: pd.DataFrame,
                        ncaa_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each WNBA-labeled player, find their college season in ncaa_df.
    Matching: fuzzy name similarity (≥0.80) within ±2 years of season_year.
    Returns a DataFrame with college features + archetype label.
    """
    ncaa_df = ncaa_df.copy()
    ncaa_df["_clean"] = ncaa_df["player"].apply(_clean_name)

    # Load draft history to get draft year per player (for tighter year matching)
    draft_year_map = {}
    if DRAFT_CSV.exists():
        draft_df = pd.read_csv(DRAFT_CSV)
        for _, row in draft_df.iterrows():
            name = _clean_name(str(row.get("player", "")))
            year = row.get("draft_year")
            if name and pd.notna(year):
                draft_year_map[name] = int(year)

    records = []
    wnba_names_seen = set()

    # Use each player's highest-MPG season as the canonical label.
    # Low-minute seasons produce noisy archetypes (e.g. Rae Burrell was
    # "3-and-D Wing" at 11 MPG in 2023, but "Combo Guard" at 16 MPG in 2024).
    # Sorting by MPG descending ensures the most-played, most-reliable season
    # wins the deduplication instead of the chronologically first one.
    wnba_df = wnba_df.sort_values("mpg", ascending=False)

    for _, wnba_row in wnba_df.iterrows():
        wnba_name  = _clean_name(str(wnba_row.get("player", "")))
        archetype  = wnba_row.get("archetype")
        wnba_year  = int(wnba_row.get("season_year", 0))

        if not wnba_name or not archetype:
            continue

        # Deduplicate: if we already have a college match for this player, skip
        if wnba_name in wnba_names_seen:
            continue

        # Expected pre-draft college season: 1–2 years before draft year.
        # If draft year not known, estimate from wnba_year (players typically
        # enter WNBA 1–2 years after their last college season).
        draft_yr  = draft_year_map.get(wnba_name, wnba_year)
        target_yr = draft_yr - 1   # last college season

        best_row  = None
        best_sim  = 0.0

        for offset in [0, 1, 2, -1]:
            yr = target_yr + offset
            candidates = ncaa_df[
                ncaa_df.get("most_recent_year", pd.Series()).eq(yr)
            ] if "most_recent_year" in ncaa_df.columns else ncaa_df

            if candidates.empty:
                continue

            sims = candidates["_clean"].apply(lambda n: _similarity(wnba_name, n))
            idx  = sims.idxmax()
            sim  = sims[idx]
            if sim > best_sim:
                best_sim = sim
                best_row = candidates.loc[idx]
            if best_sim >= 0.90:
                break

        if best_row is None or best_sim < 0.80:
            continue

        rec = {"archetype": archetype, "name_match_score": round(best_sim, 3)}
        for feat in CLASSIFIER_FEATURES:
            rec[feat] = best_row.get(feat, np.nan)
        rec["ncaa_player"]  = best_row.get("player", "")
        rec["wnba_player"]  = wnba_row.get("player", "")
        rec["match_year"]   = best_row.get("most_recent_year", np.nan)

        records.append(rec)
        wnba_names_seen.add(wnba_name)

    train_df = pd.DataFrame(records)
    return train_df

# ---------------------------------------------------------------------------
# Step 2: Train logistic regression classifier
# ---------------------------------------------------------------------------

def train_classifier(train_df: pd.DataFrame):
    """
    Train a multinomial logistic regression to predict WNBA archetype from
    college stats + height. Returns (pipeline, archetype_classes).
    """
    feat_cols = [f for f in CLASSIFIER_FEATURES if f in train_df.columns]
    X_raw = _build_feature_matrix(train_df, feat_cols)
    y     = train_df["archetype"].values

    print(f"  Training on {len(train_df)} matched players × {len(feat_cols)} features")
    print(f"  Class distribution:")
    for arch, cnt in pd.Series(y).value_counts().items():
        print(f"    {arch:20s}: {cnt}")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",  # prevents majority class (Floor General) from dominating
        )),
    ])
    pipe.fit(X_raw.values, y)

    # Print coefficients per archetype — shows which college stats drive each role
    clf        = pipe.named_steps["clf"]
    scaler     = pipe.named_steps["scaler"]
    classes    = clf.classes_
    coef_df    = pd.DataFrame(
        clf.coef_, index=classes, columns=feat_cols
    )
    print("\n  Logistic regression coefficients (top 3 per archetype):")
    for arch in classes:
        top3 = coef_df.loc[arch].abs().nlargest(3).index.tolist()
        vals = [(f, round(float(coef_df.loc[arch, f]), 3)) for f in top3]
        print(f"    {arch:20s}: {vals}")

    return pipe, feat_cols, list(classes)

# ---------------------------------------------------------------------------
# Step 3: Score (archetype_score) within each archetype
# ---------------------------------------------------------------------------

def _archetype_score(X_scaled: np.ndarray, feat_cols: list,
                      labels: np.ndarray, clf_pipe) -> np.ndarray:
    """
    Score each player on how strongly they fit their assigned archetype,
    using the classifier's predicted probability for their assigned class.
    Normalised to 0–100 within each archetype.
    """
    clf       = clf_pipe.named_steps["clf"]
    classes   = list(clf.classes_)
    proba     = clf.predict_proba(X_scaled)   # shape (n, n_classes)

    scores = np.zeros(len(labels))
    for arch in classes:
        class_idx = classes.index(arch)
        mask      = labels == arch
        p         = proba[mask, class_idx]
        lo, hi    = p.min(), p.max()
        if hi > lo:
            scores[mask] = (p - lo) / (hi - lo) * 100
        else:
            scores[mask] = 50.0
    return scores

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Step 06 — Archetype Classifier (College + Height → WNBA Role)")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if not WNBA_ARCH_CSV.exists():
        raise FileNotFoundError(
            f"{WNBA_ARCH_CSV} not found. Run 05_scrape_wnba_archetypes.py first."
        )
    if not FEAT_CSV.exists():
        raise FileNotFoundError(
            f"{FEAT_CSV} not found. Run 04_build_features.py first."
        )

    wnba_df = pd.read_csv(WNBA_ARCH_CSV)
    ncaa_df = pd.read_csv(FEAT_CSV, low_memory=False)
    print(f"  WNBA archetypes: {len(wnba_df)} player-seasons, "
          f"{wnba_df['archetype'].nunique()} archetypes")
    print(f"  NCAA players: {len(ncaa_df)}")

    # ------------------------------------------------------------------
    # Encode position for college players (derived column)
    # ------------------------------------------------------------------
    ncaa_df["pos_encoded"] = _encode_pos(ncaa_df["pos"])

    # ------------------------------------------------------------------
    # Split college players into classify-eligible vs bench
    # ------------------------------------------------------------------
    ncaa_df["_mpg_num"] = pd.to_numeric(ncaa_df.get("mpg"), errors="coerce")
    eligible_mask = ncaa_df["_mpg_num"] >= MIN_MPG_CLUSTER
    df_eligible   = ncaa_df[eligible_mask].copy().reset_index(drop=True)
    df_bench      = ncaa_df[~eligible_mask].copy().reset_index(drop=True)
    print(f"  Eligible for classification: {len(df_eligible)} "
          f"(MPG≥{MIN_MPG_CLUSTER}), bench: {len(df_bench)}")

    # ------------------------------------------------------------------
    # Build training set
    # ------------------------------------------------------------------
    print("\n  Building training set (matching WNBA players to college stats)...")
    train_df = build_training_set(wnba_df, ncaa_df)

    if len(train_df) < 20:
        raise RuntimeError(
            f"Only {len(train_df)} training examples matched. "
            "Check that WNBA draft history is scraped (run step 07) and "
            "that wnba_player_archetypes.csv is populated."
        )

    train_path = PROC_DIR / "archetype_training_set.csv"
    train_df.to_csv(train_path, index=False)
    print(f"  Training set: {len(train_df)} matched players → {train_path.name}")

    # ------------------------------------------------------------------
    # Train classifier
    # ------------------------------------------------------------------
    print("\n  Training logistic regression classifier...")
    clf_pipe, feat_cols, classes = train_classifier(train_df)

    joblib.dump(clf_pipe, MODEL_DIR / "archetype_classifier.pkl")
    print(f"\n  Model saved → data/models/archetype_classifier.pkl")

    # ------------------------------------------------------------------
    # Apply classifier to eligible college prospects
    # ------------------------------------------------------------------
    print("\n  Classifying college prospects...")
    X_eligible_raw = _build_feature_matrix(df_eligible, feat_cols)

    imputer = clf_pipe.named_steps["imputer"]
    scaler  = clf_pipe.named_steps["scaler"]
    clf_obj = clf_pipe.named_steps["clf"]

    X_imp    = imputer.transform(X_eligible_raw.values)
    X_scaled = scaler.transform(X_imp)

    labels   = clf_pipe.predict(X_eligible_raw.values)
    scores   = _archetype_score(X_scaled, feat_cols, labels, clf_pipe)

    df_eligible["archetype"]       = labels
    df_eligible["archetype_score"] = scores
    df_eligible["cluster"]         = pd.Categorical(labels, categories=classes).codes

    df_eligible["rank_in_archetype"] = (
        df_eligible.groupby("archetype")["archetype_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # PCA for webapp scatter (fit on eligible players)
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    df_eligible["pca1"] = X_2d[:, 0]
    df_eligible["pca2"] = X_2d[:, 1]
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.round(3)}  "
          f"total={pca.explained_variance_ratio_.sum():.3f}")

    # Bench players get NaN archetype columns
    for col in ["cluster", "archetype", "archetype_score", "rank_in_archetype", "pca1", "pca2"]:
        df_bench[col] = np.nan

    # ------------------------------------------------------------------
    # Archetype summary
    # ------------------------------------------------------------------
    print("\n  Archetype breakdown (eligible players only):")
    summary = df_eligible.groupby("archetype").agg(
        players=("player", "count"),
        avg_mpg=("mpg", "mean"),
        avg_pts=("pts_per_g", "mean"),
        avg_blk=("blk_per_g", "mean"),
        avg_reb=("treb_per_g", "mean"),
    ).round(2)
    print(summary.to_string())

    # ------------------------------------------------------------------
    # Recombine and write features CSV
    # ------------------------------------------------------------------
    out_df = pd.concat([df_eligible, df_bench], ignore_index=True)
    out_df = out_df.drop(columns=["_mpg_num"], errors="ignore")

    out_df.to_csv(FEAT_CSV, index=False)
    print(f"\n  Updated {FEAT_CSV.name} with archetype labels")

    # ------------------------------------------------------------------
    # Write ncaaw_players_with_archetypes_ranked.csv
    # ------------------------------------------------------------------
    # games_played
    if "games_played_latest" in out_df.columns:
        out_df["games_played"] = (
            pd.to_numeric(out_df["games_played_latest"], errors="coerce")
            .fillna(
                pd.to_numeric(out_df.get("total_mp"), errors="coerce") /
                pd.to_numeric(out_df.get("mpg"), errors="coerce").replace(0, np.nan)
            )
            .round().fillna(0).astype(int)
        )
    elif "total_mp" in out_df.columns and "mpg" in out_df.columns:
        out_df["games_played"] = (
            pd.to_numeric(out_df["total_mp"], errors="coerce") /
            pd.to_numeric(out_df.get("mpg"), errors="coerce").replace(0, np.nan)
        ).round().fillna(0).astype(int)
    else:
        out_df["games_played"] = 0

    keep_cols = [
        "player_id", "player", "pos", "most_recent_year", "first_season",
        "cluster", "archetype", "archetype_score", "rank_in_archetype",
        "pca1", "pca2",
        "pts_per_g", "ast_per_g", "treb_per_g", "blk_per_g", "stl_per_g",
        "ts_pct", "usg_pct", "bpm", "ws_per_40", "per",
        "mpg", "games_played", "readiness_score",
        "opp_win_pct",
        "height_in", "weight_lbs",
        "pts_per_g_latest", "ast_per_g_latest", "treb_per_g_latest",
        "blk_per_g_latest", "stl_per_g_latest", "ts_pct_latest",
        "fg3_pct_latest", "ft_pct_latest", "oreb_per_g_latest", "dreb_per_g_latest",
        "fg_pct_latest", "usg_pct_latest", "bpm_latest", "per_latest",
        "ws_per_40_latest", "tov_per_g_latest",
        "mpg_latest", "games_played_latest",
        "wins_latest", "losses_latest",
        "birth_year",
        "conference", "team",
    ]
    keep_cols = [c for c in keep_cols if c in out_df.columns]
    ranked_df = out_df[keep_cols].copy().rename(columns={"player": "name"})
    ranked_df.to_csv(OUT_ROOT, index=False)
    print(f"  Wrote {OUT_ROOT.name} ({len(ranked_df)} players, "
          f"{ranked_df['archetype'].nunique()} archetypes)")

    years = sorted(out_df["most_recent_year"].dropna().unique().astype(int))
    print(f"  Draft classes covered: {years}")
    print("\nDone.")


if __name__ == "__main__":
    run()
