"""
05_archetype_clusters.py
========================
Clusters all NCAAW players (all years) into 4 archetypes using KMeans, then
auto-labels each cluster by scoring its centroid profile against known archetype
signatures. Updates archetype columns in ncaaw_players_features.csv and writes
ncaaw_players_with_archetypes_ranked.csv to the project root.

Run after 04_build_features.py.

Outputs:
    data/processed/ncaaw_players_features.csv   (archetype columns updated)
    ncaaw_players_with_archetypes_ranked.csv    (project root, used by prepare_data.py)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
FEAT_CSV = ROOT / "data" / "processed" / "ncaaw_players_features.csv"
OUT_ROOT = ROOT / "ncaaw_players_with_archetypes_ranked.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K = 4   # fixed: Primary Creator, Interior Defender, Balanced Contributor, Support Player

CLUSTER_FEATURES = [
    # Usage & Scoring
    "pts_per_g",
    "fga_per_g",
    "usg_pct",
    # Shooting profile
    "ts_pct",
    "fg3_pct",
    "fg3a_per_g",
    "fta_per_g",
    # Playmaking
    "ast_per_g",
    "ast_pct",
    "tov_per_g",
    # Rebounding
    "treb_per_g",
    "treb_pct",
    # Defense
    "dws",
    "stl_per_g",
    "blk_per_g",
]

# Archetype signature vectors in standardized feature space.
# Each value is a weight: positive = cluster centroid should be above average,
# negative = below average. The 4th archetype (Balanced Contributor) is assigned
# to whichever cluster isn't claimed by the first three.
ARCHETYPE_SIGNATURES = {
    "Primary Creator": {
        "pts_per_g":   2.0,
        "ast_per_g":   1.5,
        "usg_pct":     1.5,
        "fg3a_per_g":  0.5,
        "ts_pct":      0.5,
    },
    "Interior Defender": {
        "blk_per_g":   2.0,
        "treb_per_g":  1.5,
        "treb_pct":    1.0,
        "dws":         1.0,
        "fg3a_per_g": -1.0,
    },
    "Support Player": {
        "ast_pct":     1.5,
        "stl_per_g":   1.0,
        "usg_pct":    -1.0,
        "pts_per_g":  -0.5,
        "tov_per_g":  -0.5,
    },
    # "Balanced Contributor" is the residual cluster
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _archetype_signature_score(centroid: np.ndarray, feature_names: list, signature: dict) -> float:
    """Dot product of a cluster centroid (standardized) against a signature vector."""
    score = 0.0
    for feat, weight in signature.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            score += centroid[idx] * weight
    return score


def _auto_label_clusters(centroids: np.ndarray, feature_names: list) -> dict:
    """
    Greedily assign archetype names to cluster indices.
    Each named archetype claims the cluster with the highest signature score
    (as long as it hasn't already been claimed). The remaining cluster becomes
    'Balanced Contributor'.
    """
    n_clusters = centroids.shape[0]
    assigned = {}   # archetype_name -> cluster_idx
    remaining = set(range(n_clusters))

    for archetype, signature in ARCHETYPE_SIGNATURES.items():
        scores = {
            idx: _archetype_signature_score(centroids[idx], feature_names, signature)
            for idx in remaining
        }
        best = max(scores, key=scores.get)
        assigned[archetype] = best
        remaining.remove(best)
        print(f"  {archetype:22s} → cluster {best}  (score {scores[best]:.3f})")

    for idx in remaining:
        assigned["Balanced Contributor"] = idx
        print(f"  {'Balanced Contributor':22s} → cluster {idx}  (residual)")

    # Return cluster_idx -> archetype_name
    return {v: k for k, v in assigned.items()}


def _archetype_score_matrix(X_scaled: np.ndarray, feature_names: list,
                             cluster_labels: np.ndarray,
                             cluster_to_archetype: dict) -> np.ndarray:
    """
    Score each player on how strongly they embody their assigned archetype using
    the dot product of their scaled stats against the archetype signature vector.
    Exceptional players (e.g. elite blockers for Interior Defender) score higher
    than average players. Normalised to 0–100 within each archetype.
    """
    scores = np.zeros(len(X_scaled))
    for cluster_idx, archetype in cluster_to_archetype.items():
        signature = ARCHETYPE_SIGNATURES.get(archetype, {})
        if not signature:
            # Balanced Contributor: use overall magnitude (how impactful they are)
            sig_vec = np.ones(len(feature_names))
        else:
            sig_vec = np.array([signature.get(f, 0.0) for f in feature_names])

        mask = cluster_labels == cluster_idx
        dot_scores = X_scaled[mask] @ sig_vec
        lo, hi = dot_scores.min(), dot_scores.max()
        if hi > lo:
            scores[mask] = (dot_scores - lo) / (hi - lo) * 100
        else:
            scores[mask] = 50.0
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Step 05 — Archetype Clustering")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------
    if not FEAT_CSV.exists():
        raise FileNotFoundError(f"Features CSV not found: {FEAT_CSV}\nRun 04_build_features.py first.")

    df = pd.read_csv(FEAT_CSV)
    print(f"  Loaded {len(df)} players from {FEAT_CSV.name}")

    # ------------------------------------------------------------------
    # Validate / coerce feature columns
    # ------------------------------------------------------------------
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    def _build_feature_matrix(src_df, cols, pct_cols):
        X = src_df[cols].copy()
        for c in cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        for c in pct_cols:
            if c in X.columns:
                med = np.nanmedian(X[c].values)
                if med > 1.5:
                    X[c] = X[c] / 100.0
        return X

    pct_cols = {"usg_pct", "ts_pct", "fg3_pct", "ast_pct", "treb_pct"}

    # Weighted multi-year averages — used for cluster assignment and PCA
    X_raw = _build_feature_matrix(df, CLUSTER_FEATURES, pct_cols)

    # Latest-season stats — used for archetype_score so ranking reflects current form.
    # Falls back to the weighted column when the _latest column is missing or NaN.
    latest_cols = {c: f"{c}_latest" for c in CLUSTER_FEATURES}
    X_latest_src = df.copy()
    for base, latest in latest_cols.items():
        if latest in df.columns:
            filled = pd.to_numeric(df[latest], errors="coerce").fillna(
                pd.to_numeric(df[base], errors="coerce")
            )
        else:
            filled = pd.to_numeric(df[base], errors="coerce")
        X_latest_src[base] = filled
    X_latest_raw = _build_feature_matrix(X_latest_src, CLUSTER_FEATURES, pct_cols)

    # ------------------------------------------------------------------
    # Fit clustering pipeline (uses weighted averages for stable assignments)
    # ------------------------------------------------------------------
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("kmeans",  KMeans(n_clusters=K, random_state=42, n_init="auto")),
    ])

    print(f"\n  Fitting KMeans (K={K}) on {len(df)} players × {len(CLUSTER_FEATURES)} features...")
    labels = pipe.fit_predict(X_raw)

    # Scaled weighted matrix (for PCA)
    X_imp    = pipe.named_steps["imputer"].transform(X_raw)
    X_scaled = pipe.named_steps["scaler"].transform(X_imp)
    centroids_scaled = pipe.named_steps["kmeans"].cluster_centers_

    # Scaled latest-season matrix (for archetype_score)
    # Reuse the same imputer/scaler so scores are comparable
    X_latest_imp    = pipe.named_steps["imputer"].transform(X_latest_raw)
    X_latest_scaled = pipe.named_steps["scaler"].transform(X_latest_imp)

    print(f"\n  Cluster sizes: { {i: int((labels==i).sum()) for i in range(K)} }")

    # ------------------------------------------------------------------
    # Auto-label clusters
    # ------------------------------------------------------------------
    print("\n  Auto-labeling clusters:")
    cluster_to_archetype = _auto_label_clusters(centroids_scaled, CLUSTER_FEATURES)

    # ------------------------------------------------------------------
    # PCA (2-D, for visualization / webapp scatter)
    # ------------------------------------------------------------------
    pca   = PCA(n_components=2, random_state=42)
    X_2d  = pca.fit_transform(X_scaled)
    print(f"\n  PCA explained variance: {pca.explained_variance_ratio_.round(3)}  "
          f"total={pca.explained_variance_ratio_.sum():.3f}")

    # ------------------------------------------------------------------
    # Archetype scores and ranks
    # ------------------------------------------------------------------
    df["cluster"]   = labels
    df["archetype"] = df["cluster"].map(cluster_to_archetype)
    df["pca1"]      = X_2d[:, 0]
    df["pca2"]      = X_2d[:, 1]

    # archetype_score: based on latest-season stats so ranking reflects current
    # form, not a career average diluted by older or injury-shortened seasons.
    df["archetype_score"] = _archetype_score_matrix(
        X_latest_scaled, CLUSTER_FEATURES, labels, cluster_to_archetype
    )

    # rank_in_archetype: ranked across all years (used by XGBoost training and
    # re-ranked to 2025-only in prepare_data.py)
    df["rank_in_archetype"] = (
        df.groupby("archetype")["archetype_score"]
          .rank(ascending=False, method="min")
          .astype(int)
    )

    # ------------------------------------------------------------------
    # Print archetype summary
    # ------------------------------------------------------------------
    print("\n  Archetype breakdown:")
    summary = df.groupby("archetype").agg(
        players=("player", "count"),
        avg_pts=("pts_per_g", "mean"),
        avg_ast=("ast_per_g", "mean"),
        avg_blk=("blk_per_g", "mean"),
        avg_reb=("treb_per_g", "mean"),
    ).round(2)
    print(summary.to_string())

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    # 1. Update ncaaw_players_features.csv
    arch_cols = ["cluster", "archetype", "archetype_score", "rank_in_archetype", "pca1", "pca2"]
    for col in arch_cols:
        # Drop old column if present so we overwrite cleanly
        if col in df.columns and col not in [c for c in df.columns if c == col]:
            pass  # already updated in-place above

    df.to_csv(FEAT_CSV, index=False)
    print(f"\n  Updated {FEAT_CSV.name} with archetype labels ({len(df)} players)")

    # games_played: use the latest-season value from 03_build_features when available;
    # fall back to deriving from total_mp / mpg (multi-year total) only as a last resort.
    if "games_played_latest" in df.columns:
        df["games_played"] = (
            pd.to_numeric(df["games_played_latest"], errors="coerce")
            .fillna(
                pd.to_numeric(df.get("total_mp"), errors="coerce") /
                pd.to_numeric(df.get("mpg"), errors="coerce").replace(0, np.nan)
            )
            .round().fillna(0).astype(int)
        )
    elif "total_mp" in df.columns and "mpg" in df.columns:
        df["games_played"] = (
            pd.to_numeric(df["total_mp"], errors="coerce") /
            pd.to_numeric(df["mpg"], errors="coerce").replace(0, np.nan)
        ).round().fillna(0).astype(int)
    else:
        df["games_played"] = 0

    # 2. Write ncaaw_players_with_archetypes_ranked.csv (project root)
    keep_cols = [
        "player_id", "player", "pos", "most_recent_year",
        "cluster", "archetype", "archetype_score", "rank_in_archetype",
        "pca1", "pca2",
        # Weighted multi-year averages (used by model)
        "pts_per_g", "ast_per_g", "treb_per_g", "blk_per_g", "stl_per_g",
        "ts_pct", "usg_pct", "bpm", "ws_per_40", "per",
        "mpg", "games_played", "readiness_score",
        "opp_win_pct",
        # Latest-season stats (for display in webapp)
        "pts_per_g_latest", "ast_per_g_latest", "treb_per_g_latest",
        "blk_per_g_latest", "stl_per_g_latest", "ts_pct_latest",
        "fg3_pct_latest", "ft_pct_latest", "oreb_per_g_latest", "dreb_per_g_latest",
        "fg_pct_latest", "usg_pct_latest", "bpm_latest", "per_latest",
        "ws_per_40_latest", "tov_per_g_latest",
        "mpg_latest", "games_played_latest",
        "conference", "team",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out_df = df[keep_cols].copy()
    out_df = out_df.rename(columns={"player": "name"})
    out_df.to_csv(OUT_ROOT, index=False)
    print(f"  Wrote {OUT_ROOT.name} ({len(out_df)} players, {len(out_df['archetype'].unique())} archetypes)")

    years = sorted(df["most_recent_year"].dropna().unique().astype(int))
    print(f"  Draft classes covered: {years}")
    print("\nDone.")


if __name__ == "__main__":
    run()
