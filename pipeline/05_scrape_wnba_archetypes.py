"""
05_scrape_wnba_archetypes.py
=============================
Scrapes WNBA player season stats from basketball-reference and clusters
WNBA players into archetypes using GMM. These archetype labels become
the ground-truth training signal for the college → WNBA archetype
classifier in step 06.

Why cluster WNBA players first?
  WNBA archetypes don't come pre-labeled. By clustering players on their
  actual WNBA stats (not college stats), we get role labels that reflect
  what these players became in the pros — not what they looked like in college.
  Step 06 then learns which college stats + height predict those pro roles.

Run after 04_build_features.py.

Outputs:
    data/raw/wnba_player_season_stats.csv      (scraped, cached)
    data/processed/wnba_player_archetypes.csv  (with archetype labels)
"""

import re
import time
import random
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from bs4 import BeautifulSoup, Comment

from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import cloudscraper
    _HAS_CLOUDSCRAPER = True
except ImportError:
    _HAS_CLOUDSCRAPER = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

WNBA_SEASONS   = list(range(2019, 2026))  # 2019–2025
MIN_MPG        = 10.0
MIN_GAMES      = 10
K              = 6   # archetypes — must match step 06

MAX_RETRIES    = 5
SLEEP_RANGE    = (4.0, 7.0)

# Features used to cluster WNBA players — mirrors the college clustering
# features so that GMM components are interpretable in the same role space.
CLUSTER_FEATURES = [
    "three_rate",    # 3PA / FGA
    "inside_score",  # fg2 * fg2_pct — paint scoring
    "fta_per_g",
    "fg3_pct",
    "treb_pct",
    "oreb_pct",
    "blk_per_g",
    "usg_pct",
    "ast_pct",
    "tov_per_g",
    "pos_encoded",
]

_POS_ENCODING = {
    "G": 0.0, "G-F": 0.25, "F-G": 0.25,
    "F": 0.5, "F-C": 0.75, "C-F": 0.75, "C": 1.0,
}

ARCHETYPE_SIGNATURES = {
    "Floor General": {
        "ast_pct":      2.5,
        "usg_pct":      1.5,
        "tov_per_g":    0.5,
        "three_rate":   0.5,
        "pos_encoded": -2.5,
    },
    "Post Scorer": {
        "usg_pct":      2.0,
        "fta_per_g":    1.5,
        "inside_score": 1.5,
        "treb_pct":     1.0,
        "pos_encoded":  2.0,
    },
    "Combo Guard": {
        "usg_pct":      1.0,
        "three_rate":   1.0,
        "ast_pct":      1.0,
        "fta_per_g":    0.5,
        "pos_encoded": -1.5,
    },
    "3-and-D Wing": {
        "three_rate":   2.5,
        "fg3_pct":      2.0,
        "pos_encoded":  0.5,
    },
    "Stretch Big": {
        "treb_pct":     1.5,
        "oreb_pct":     0.5,
        "three_rate":   1.5,
        "fg3_pct":      1.0,
        "pos_encoded":  2.0,
    },
    "Interior Big": {
        "blk_per_g":    2.0,
        "treb_pct":     2.0,
        "oreb_pct":     1.5,
        "inside_score": 1.0,
        "pos_encoded":  2.5,
    },
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def make_session():
    if _HAS_CLOUDSCRAPER:
        return cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "mobile": False}
        )
    return requests.Session()


def fetch_html(url: str, session) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed {url}: {e}")
            time.sleep(10.0 * attempt)
    return ""


def soups_with_comments(html: str):
    soup = BeautifulSoup(html, "html.parser")
    result = [soup]
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            result.append(BeautifulSoup(c, "html.parser"))
    return result


def find_table(soups, table_id: str):
    for s in soups:
        t = s.find("table", id=table_id)
        if t:
            return t
    return None


def table_to_df(table) -> pd.DataFrame:
    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()
    rows = []
    for tr in tbody.find_all("tr"):
        cls = tr.get("class") or []
        if "thead" in cls or "partial_table" in cls:
            continue
        row = {}
        for stat in ("name_display", "player"):
            td = tr.find("td", attrs={"data-stat": stat})
            if td:
                a = td.find("a", href=True)
                if a:
                    # Use the anchor text — td.get_text() can pull the whole row
                    name = a.get_text(strip=True)
                    row["player_href"] = a["href"].strip()
                else:
                    name = td.get_text(strip=True)
                if name:
                    row["player"] = name
                    break
        if "player" not in row:
            th = tr.find("th")
            if th:
                a = th.find("a", href=True)
                if a:
                    row["player"] = a.get_text(strip=True)
                    row["player_href"] = a["href"].strip()
                else:
                    row["player"] = th.get_text(strip=True)
        for td in tr.find_all("td"):
            ds = td.get("data-stat")
            if ds and ds not in ("name_display", "player"):
                row[ds] = td.get_text(strip=True)
        if row and row.get("player"):
            rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def scrape_season(year: int, session) -> pd.DataFrame:
    """Scrape per-game + advanced stats for one WNBA season, return merged df."""
    per_url = f"https://www.basketball-reference.com/wnba/years/{year}_per_game.html"
    adv_url = f"https://www.basketball-reference.com/wnba/years/{year}_advanced.html"

    print(f"  {year}: per-game...", end=" ", flush=True)
    try:
        per_html  = fetch_html(per_url, session)
        per_soups = soups_with_comments(per_html)
        per_table = find_table(per_soups, "per_game_stats") or find_table(per_soups, "per_game")
        per_df    = table_to_df(per_table) if per_table else pd.DataFrame()
        print(f"{len(per_df)} rows", end=" | ", flush=True)
    except Exception as e:
        print(f"FAIL ({e})")
        return pd.DataFrame()

    time.sleep(random.uniform(*SLEEP_RANGE))

    print(f"advanced...", end=" ", flush=True)
    try:
        adv_html  = fetch_html(adv_url, session)
        adv_soups = soups_with_comments(adv_html)
        adv_table = find_table(adv_soups, "advanced_stats") or find_table(adv_soups, "advanced")
        adv_df    = table_to_df(adv_table) if adv_table else pd.DataFrame()
        print(f"{len(adv_df)} rows")
    except Exception as e:
        print(f"FAIL ({e})")
        adv_df = pd.DataFrame()

    time.sleep(random.uniform(*SLEEP_RANGE))

    if per_df.empty:
        return pd.DataFrame()

    # Numeric coerce
    num_per = ["g", "mp", "mp_per_g", "fg_per_g", "fga_per_g", "fg3_per_g",
               "fg3a_per_g", "fg3_pct", "fg2_per_g", "fg2a_per_g", "fg2_pct",
               "ft_per_g", "fta_per_g", "ft_pct", "orb_per_g", "drb_per_g",
               "trb_per_g", "ast_per_g", "stl_per_g", "blk_per_g", "tov_per_g", "pts_per_g"]
    num_adv = ["ts_pct", "fg3a_per_fga_pct", "fta_per_fga_pct", "orb_pct",
               "drb_pct", "trb_pct", "ast_pct", "stl_pct", "blk_pct",
               "tov_pct", "usg_pct", "ws_per_40"]

    for col in num_per:
        if col in per_df.columns:
            per_df[col] = pd.to_numeric(per_df[col], errors="coerce")
    for col in num_adv:
        if col in adv_df.columns:
            adv_df[col] = pd.to_numeric(adv_df[col], errors="coerce")

    # Drop totals / separator rows
    per_df = per_df[per_df["player"].str.strip().str.len() > 1].copy()
    per_df = per_df[pd.to_numeric(per_df.get("g", pd.Series()), errors="coerce").notna()].copy()

    per_df["season_year"] = year

    # Merge advanced
    if not adv_df.empty and "player" in adv_df.columns:
        adv_df = adv_df[adv_df["player"].str.strip().str.len() > 1].copy()
        adv_keep = ["player"] + [c for c in num_adv if c in adv_df.columns]
        # Some players appear twice (traded mid-season) — keep first occurrence
        adv_df = adv_df[adv_keep].drop_duplicates(subset=["player"])
        merged = per_df.merge(adv_df, on="player", how="left", suffixes=("", "_adv"))
    else:
        merged = per_df

    return merged


def load_or_scrape_wnba_stats(session) -> pd.DataFrame:
    cache = RAW_DIR / "wnba_player_season_stats.csv"
    if cache.exists():
        df = pd.read_csv(cache, low_memory=False)
        cached_years = set(df["season_year"].unique()) if "season_year" in df.columns else set()
        missing = [y for y in WNBA_SEASONS if y not in cached_years]
        if not missing:
            print(f"  [CACHE] {len(df)} rows from {cache.name}")
            return df
        print(f"  Fetching missing seasons: {missing}")
        new_frames = [df]
        for year in missing:
            frame = scrape_season(year, session)
            if not frame.empty:
                new_frames.append(frame)
            time.sleep(random.uniform(*SLEEP_RANGE))
        out = pd.concat(new_frames, ignore_index=True)
        out.to_csv(cache, index=False)
        return out

    print(f"  Scraping WNBA stats for seasons {WNBA_SEASONS[0]}–{WNBA_SEASONS[-1]}...")
    frames = []
    for year in WNBA_SEASONS:
        frame = scrape_season(year, session)
        if not frame.empty:
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(cache, index=False)
    print(f"  Saved {len(out)} rows → {cache.name}")
    return out

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _encode_pos(series: pd.Series) -> pd.Series:
    def _parse(val):
        if not isinstance(val, str):
            return np.nan
        m = re.search(r'\b([GFC](?:-[GFC])?)\b', val)
        return _POS_ENCODING.get(m.group(1), np.nan) if m else np.nan
    enc = series.apply(_parse)
    return enc.fillna(0.4)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute clustering features from raw scraped columns."""
    out = df.copy()

    def _num(col):
        return pd.to_numeric(out.get(col, pd.Series(np.nan, index=out.index)), errors="coerce")

    # basketball-reference per-game tables use _per_g suffix on counting stats
    fga     = _num("fga_per_g").replace(0, np.nan)
    fg3a    = _num("fg3a_per_g")
    fg2     = _num("fg2_per_g")
    fg2_pct = _num("fg2_pct")
    g       = _num("g").replace(0, np.nan)

    out["mpg"]          = _num("mp_per_g").round(1)
    out["fta_per_g"]    = _num("fta_per_g").round(2)   # already per-game in scraped data
    out["blk_per_g"]    = _num("blk_per_g").round(2)
    out["tov_per_g"]    = _num("tov_per_g").round(2)
    out["three_rate"]   = (fg3a / fga).fillna(0).round(4)
    out["inside_score"] = (fg2 * fg2_pct.fillna(0)).round(4)
    out["fg3_pct"]      = _num("fg3_pct")

    # Advanced rate stats — bball-ref uses trb_pct/orb_pct; map to treb_pct/oreb_pct
    if "trb_pct" in out.columns and not out["trb_pct"].isna().all():
        out["treb_pct"] = pd.to_numeric(out["trb_pct"], errors="coerce")
    if "orb_pct" in out.columns and not out["orb_pct"].isna().all():
        out["oreb_pct"] = pd.to_numeric(out["orb_pct"], errors="coerce")

    for col in ["treb_pct", "oreb_pct", "ast_pct", "usg_pct"]:
        if col not in out.columns or out[col].isna().all():
            # rough estimates from per-game counting stats when advanced table is missing
            if col == "treb_pct":
                out[col] = (_num("trb_per_g") / _num("mp_per_g") * 40 / 2).clip(0, 30)
            elif col == "oreb_pct":
                out[col] = (_num("orb_per_g") / _num("mp_per_g") * 40 / 2).clip(0, 20)
            elif col == "ast_pct":
                out[col] = (_num("ast_per_g") / _num("mp_per_g") * 40 / 5).clip(0, 50)
            elif col == "usg_pct":
                out[col] = pd.to_numeric(out.get("usg_pct"), errors="coerce")

    # pct columns stored as 0–100 — normalise to 0–1
    for col in ["treb_pct", "oreb_pct", "ast_pct", "usg_pct", "fg3_pct"]:
        if col in out.columns:
            med = out[col].dropna().median()
            if med > 1.5:
                out[col] = out[col] / 100.0

    out["pos_encoded"] = _encode_pos(out.get("pos", pd.Series("", index=out.index)))

    return out

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _auto_label_clusters(centroids: np.ndarray, feature_names: list) -> dict:
    arch_names = list(ARCHETYPE_SIGNATURES.keys())
    score_matrix = np.zeros((len(centroids), len(arch_names)))
    for i, centroid in enumerate(centroids):
        for j, sig in enumerate(ARCHETYPE_SIGNATURES.values()):
            sig_vec = np.array([sig.get(f, 0.0) for f in feature_names])
            score_matrix[i, j] = float(np.dot(centroid, sig_vec))
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    assignment = {int(r): arch_names[c] for r, c in zip(row_ind, col_ind)}
    for ci, name in sorted(assignment.items()):
        j = arch_names.index(name)
        print(f"    cluster {ci} → {name:20s}  score={score_matrix[ci, j]:+.3f}")
    return assignment


def cluster_wnba_players(df: pd.DataFrame) -> pd.DataFrame:
    """Run GMM, assign archetype labels, return df with cluster/archetype columns."""
    # Filter to significant players
    mpg_ok = pd.to_numeric(df.get("mpg"), errors="coerce") >= MIN_MPG
    g_ok   = pd.to_numeric(df.get("g"),   errors="coerce") >= MIN_GAMES
    mask   = mpg_ok & g_ok
    df_fit = df[mask].copy().reset_index(drop=True)
    print(f"  Clustering {len(df_fit)} player-seasons (MPG≥{MIN_MPG}, G≥{MIN_GAMES})")

    # Validate features
    missing = [c for c in CLUSTER_FEATURES if c not in df_fit.columns]
    if missing:
        raise ValueError(f"Missing clustering features: {missing}")

    X_raw = df_fit[CLUSTER_FEATURES].copy()
    for col in CLUSTER_FEATURES:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("gmm",     GaussianMixture(
            n_components=K,
            covariance_type="full",
            n_init=15,
            max_iter=300,
            reg_covar=1e-4,
            random_state=42,
        )),
    ])

    labels = pipe.fit_predict(X_raw)
    gmm    = pipe.named_steps["gmm"]
    print(f"  GMM converged={gmm.converged_}  log-likelihood={gmm.lower_bound_:.4f}")
    print(f"  Component sizes: { {i: int((labels==i).sum()) for i in range(K)} }")

    X_scaled  = pipe.named_steps["scaler"].transform(
        pipe.named_steps["imputer"].transform(X_raw)
    )
    centroids = gmm.means_

    print("\n  Archetype assignment (Hungarian):")
    cluster_to_arch = _auto_label_clusters(centroids, CLUSTER_FEATURES)

    df_fit["cluster"]   = labels
    df_fit["archetype"] = df_fit["cluster"].map(cluster_to_arch)

    # PCA for optional visualisation
    pca      = PCA(n_components=2, random_state=42)
    X_2d     = pca.fit_transform(X_scaled)
    df_fit["pca1"] = X_2d[:, 0]
    df_fit["pca2"] = X_2d[:, 1]
    print(f"\n  PCA explained variance: {pca.explained_variance_ratio_.round(3)}  "
          f"total={pca.explained_variance_ratio_.sum():.3f}")

    return df_fit

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Step 05 — Scrape & Cluster WNBA Archetypes")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    with make_session() as session:
        raw_df = load_or_scrape_wnba_stats(session)

    if raw_df.empty:
        raise RuntimeError("No WNBA stats scraped — check network / basketball-reference.")

    print(f"\n  Building features for {len(raw_df)} player-seasons...")
    feat_df = build_features(raw_df)

    print("\n  Clustering WNBA players into archetypes...")
    clustered = cluster_wnba_players(feat_df)

    # Archetype breakdown
    print("\n  Archetype breakdown:")
    summary = (
        clustered.groupby("archetype")
        .agg(player_seasons=("player", "count"),
             avg_pts=("pts_per_g", "mean"),
             avg_ast=("ast_per_g", "mean"),
             avg_blk=("blk_per_g", "mean"),
             avg_reb=("trb_per_g", "mean"))
        .round(2)
    )
    print(summary.to_string())

    # Save
    out_path = PROC_DIR / "wnba_player_archetypes.csv"
    keep = ["player", "player_href", "season_year", "pos",
            "g", "mpg", "pts_per_g", "ast_per_g", "trb_per_g", "blk_per_g", "stl_per_g",
            ] + CLUSTER_FEATURES + ["cluster", "archetype", "pca1", "pca2"]
    keep = [c for c in keep if c in clustered.columns]
    clustered[keep].to_csv(out_path, index=False)
    print(f"\n  Saved {len(clustered)} labeled player-seasons → {out_path.name}")
    print(f"  Archetypes: {sorted(clustered['archetype'].unique())}")
    print("\nDone.")


if __name__ == "__main__":
    run()
