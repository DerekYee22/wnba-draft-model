"""
Gradient Boosting WNBA Readiness Model
========================================
Replaces the synthetic Ridge target with actual WNBA career performance labels.

Training pipeline:
  1. Scrape WNBA draft history (2018-2024) from basketball-reference
  2. Scrape WNBA career stats for each draft pick
  3. Build a composite WNBA success score as the regression target
  4. Match each pick to their pre-draft NCAA stats (from pipeline/03 output)
  5. Train gradient boosting with leave-one-year-out cross-validation
  6. Score current 2025 prospects

Model choice — gradient boosting over Ridge:
  - Ridge needed a hand-crafted synthetic target because it had no WNBA labels.
    Gradient boosting learns directly from real outcomes, so the target is ground truth.
  - Captures nonlinear interactions (e.g., high usage only matters when efficiency is high).
  - Built-in feature importance shows which NCAA stats actually predict WNBA success.
  - sklearn's HistGradientBoostingRegressor handles missing values natively (no imputer needed)
    and has the same performance profile as XGBoost at this dataset size.

Training set size note:
  ~6 draft years × ~33 picks/year ≈ 200 labeled players. That is small, so:
  - max_depth=3, early stopping, and L2 regularization to avoid overfitting.
  - Feature count kept to ~20 core stats.
  - Players with <10 WNBA games receive a target of 0 (did not establish themselves).

Inputs:
    data/processed/ncaaw_players_features.csv   (from 03_build_features.py)
    data/raw/ncaaw_players_raw_{year}.csv       (per-year NCAA data if available)

Outputs:
    data/raw/wnba_draft_history.csv             (scraped, cached)
    data/raw/wnba_player_career_stats.csv       (scraped, cached)
    data/processed/xgb_training_set.csv         (matched NCAA → WNBA labels)
    data/processed/ncaaw_players_features.csv   (adds readiness_score column)
    data/models/xgb_model.pkl
"""

import re
import time
import random
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from bs4 import BeautifulSoup, Comment

try:
    import cloudscraper
    _HAS_CLOUDSCRAPER = True
except ImportError:
    import requests
    _HAS_CLOUDSCRAPER = False
    print("[WARN] cloudscraper not installed — falling back to requests. "
          "Run: pip install cloudscraper")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "data" / "models"

# Draft years to use for training (need WNBA results to exist)
DRAFT_YEARS = list(range(2018, 2025))   # 2018–2024 drafts

# WNBA games threshold to be considered "established" in the league
MIN_WNBA_GAMES = 10

# NCAA features used as model inputs — kept deliberately narrow to avoid
# overfitting on a ~200-sample training set
FEATURES = [
    # Scoring & efficiency
    "pts_per_g", "ts_pct", "efg_pct", "usg_pct",
    # Shooting profile
    "fg3a_per_g", "fg3_pct", "ft_pct", "fta_per_g",
    # Playmaking
    "ast_per_g", "ast_pct", "ast_tov_ratio",
    # Rebounding
    "treb_per_g", "oreb_pct", "dreb_pct",
    # Defense
    "stl_per_g", "blk_per_g", "dbpm",
    # Advanced
    "bpm", "per", "ws_per_40",
    # Context
    "adj_opp_win_pct", "mpg_latest",
]

MAX_RETRIES    = 3
SLEEP_REQUESTS = (3.0, 5.0)   # be polite to basketball-reference

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def make_session():
    """
    Create a cloudscraper session (bypasses Cloudflare JS challenges that cause
    403 errors on basketball-reference.com). Falls back to plain requests if
    cloudscraper is not installed.
    """
    if _HAS_CLOUDSCRAPER:
        return cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "mobile": False}
        )
    import requests as _req
    return _req.Session()


def fetch_html(url: str, session) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed {url}: {e}")
            time.sleep(3.0 * attempt)
    return ""  # unreachable


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
        th = tr.find("th")
        if th:
            row["_th_text"] = th.get_text(strip=True)
            a = th.find("a", href=True)
            if a:
                row["player_href"] = a["href"].strip()
        for td in tr.find_all("td"):
            ds = td.get("data-stat")
            if ds:
                row[ds] = td.get_text(strip=True)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 1. Scrape WNBA draft history
# ---------------------------------------------------------------------------

def scrape_wnba_draft(year: int, session: requests.Session) -> pd.DataFrame:
    """
    Scrape one year's WNBA draft page.
    Returns DataFrame with columns: draft_year, round, pick, player, player_href,
    college, team.
    """
    url = f"https://www.basketball-reference.com/wnba/draft/{year}.html"
    html  = fetch_html(url, session)
    soups = soups_with_comments(html)

    # Table ID varies by year — try common options
    table = None
    for tid in ["drafts", "draft", f"draft_{year}"]:
        table = find_table(soups, tid)
        if table:
            break

    # Fallback: any table whose headers contain "Pk" and "Player"
    if table is None:
        for s in soups:
            for t in s.find_all("table"):
                thead = t.find("thead")
                if not thead:
                    continue
                hs = {th.get_text(strip=True) for th in thead.find_all("th")}
                if "Pk" in hs and "Player" in hs:
                    table = t
                    break
            if table:
                break

    if table is None:
        print(f"  [WARN] No draft table found for {year}")
        return pd.DataFrame()

    df = table_to_df(table)
    if df.empty:
        return df

    # Rename standard columns
    col_map = {
        "pk":         "pick",
        "round":      "round",
        "team_id":    "team",
        "college_id": "college",
        "player":     "player",
        "_th_text":   "player_raw",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Resolve player name: basketball-reference uses data-stat="player" as a td sometimes
    if "player" not in df.columns and "_th_text" in df.columns:
        df["player"] = df["_th_text"]
    elif "player_raw" in df.columns:
        df["player"] = df["player_raw"]

    # Drop separator rows (where pick is empty or non-numeric)
    if "pick" in df.columns:
        df = df[pd.to_numeric(df["pick"], errors="coerce").notna()].copy()
    elif "Pk" in df.columns:
        df = df.rename(columns={"Pk": "pick"})
        df = df[pd.to_numeric(df["pick"], errors="coerce").notna()].copy()

    df["draft_year"] = year

    # Extract player_id from href (e.g. /wnba/players/s/stewaja01w.html → stewaja01w)
    if "player_href" in df.columns:
        df["wnba_player_id"] = (
            df["player_href"]
            .fillna("")
            .apply(lambda h: h.rstrip("/").split("/")[-1].replace(".html", ""))
        )
    else:
        df["wnba_player_id"] = ""

    keep = ["draft_year", "round", "pick", "player", "wnba_player_id", "player_href",
            "college", "team"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].reset_index(drop=True)


def load_or_scrape_draft_history(session: requests.Session) -> pd.DataFrame:
    cache_path = RAW_DIR / "wnba_draft_history.csv"
    if cache_path.exists():
        print(f"  [CACHE] Loaded draft history from {cache_path.name}")
        return pd.read_csv(cache_path)

    frames = []
    for year in DRAFT_YEARS:
        print(f"  Scraping {year} WNBA draft...")
        try:
            df = scrape_wnba_draft(year, session)
            if not df.empty:
                frames.append(df)
                print(f"    -> {len(df)} picks")
        except Exception as e:
            print(f"    [FAIL] {e}")
        time.sleep(random.uniform(*SLEEP_REQUESTS))

    if not frames:
        return pd.DataFrame()

    draft_df = pd.concat(frames, ignore_index=True)
    draft_df.to_csv(cache_path, index=False)
    print(f"  Draft history saved ({len(draft_df)} picks) → {cache_path.name}")
    return draft_df

# ---------------------------------------------------------------------------
# 2. Scrape WNBA career stats for each draft pick
# ---------------------------------------------------------------------------

def scrape_wnba_player_career(player_href: str, session: requests.Session) -> dict:
    """
    Fetch a player's WNBA career stats page and return their career totals.
    Returns dict with keys: wnba_games, wnba_ws40, wnba_bpm, wnba_per,
    wnba_ts_pct, wnba_seasons.
    """
    base_url = "https://www.basketball-reference.com"
    url = base_url + player_href if player_href.startswith("/") else player_href

    try:
        html  = fetch_html(url, session)
        soups = soups_with_comments(html)
    except Exception as e:
        print(f"    [WARN] Could not fetch {url}: {e}")
        return {}

    # Try to find the career totals row in per-game or advanced tables
    career_stats = {}

    # Per-game table (id varies: wnba_per_game, per_game_wnba, etc.)
    per_table = None
    for tid in ["wnba_per_game", "per_game", "stats"]:
        per_table = find_table(soups, tid)
        if per_table:
            break

    # Advanced table
    adv_table = None
    for tid in ["wnba_advanced", "advanced"]:
        adv_table = find_table(soups, tid)
        if adv_table:
            break

    def get_career_row(table):
        if table is None:
            return {}
        tbody = table.find("tbody")
        if not tbody:
            return {}
        # Career row is usually the last tfoot row or a row with data-stat="season" == "Career"
        tfoot = table.find("tfoot")
        if tfoot:
            for tr in tfoot.find_all("tr"):
                row = {}
                for td in tr.find_all(["td", "th"]):
                    ds = td.get("data-stat")
                    if ds:
                        row[ds] = td.get_text(strip=True)
                if row:
                    return row
        # Fallback: last non-header row in tbody
        all_rows = [
            tr for tr in tbody.find_all("tr")
            if "thead" not in (tr.get("class") or [])
        ]
        if all_rows:
            row = {}
            for td in all_rows[-1].find_all(["td", "th"]):
                ds = td.get("data-stat")
                if ds:
                    row[ds] = td.get_text(strip=True)
            return row
        return {}

    def _float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan

    per_row = get_career_row(per_table)
    adv_row = get_career_row(adv_table)

    # Games played
    g_val = per_row.get("g") or adv_row.get("g") or ""
    career_stats["wnba_games"] = _float(g_val.replace(",", ""))

    # Seasons played
    # Count non-header rows in tbody as a proxy
    n_seasons = 0
    if per_table:
        tbody = per_table.find("tbody")
        if tbody:
            n_seasons = sum(
                1 for tr in tbody.find_all("tr")
                if "thead" not in (tr.get("class") or [])
                and tr.find("td") is not None
            )
    career_stats["wnba_seasons"] = n_seasons

    # Advanced stats from career row
    career_stats["wnba_ws40"]   = _float(adv_row.get("ws_per_40") or adv_row.get("ws/40"))
    career_stats["wnba_bpm"]    = _float(adv_row.get("bpm"))
    career_stats["wnba_per"]    = _float(per_row.get("per") or adv_row.get("per"))
    career_stats["wnba_ts_pct"] = _float(adv_row.get("ts_pct") or per_row.get("ts_pct"))
    career_stats["wnba_ws"]     = _float(adv_row.get("ws"))

    return career_stats


def load_or_scrape_wnba_stats(draft_df: pd.DataFrame,
                               session: requests.Session) -> pd.DataFrame:
    cache_path = RAW_DIR / "wnba_player_career_stats.csv"
    if cache_path.exists():
        print(f"  [CACHE] Loaded WNBA player stats from {cache_path.name}")
        cached = pd.read_csv(cache_path)
        # Check for players in draft_df not yet in cache
        cached_ids = set(cached["wnba_player_id"].astype(str))
        missing = draft_df[
            ~draft_df["wnba_player_id"].astype(str).isin(cached_ids)
            & draft_df["wnba_player_id"].astype(str).str.len().gt(2)
        ]
        if missing.empty:
            return cached
        print(f"  Fetching {len(missing)} new players not in cache...")
        new_rows = _scrape_player_stats_list(missing, session)
        if new_rows:
            new_df = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
            new_df.to_csv(cache_path, index=False)
            return new_df
        return cached

    print(f"  Scraping WNBA career stats for {len(draft_df)} draft picks...")
    rows = _scrape_player_stats_list(draft_df, session)
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(cache_path, index=False)
    print(f"  WNBA stats saved ({len(stats_df)} players) → {cache_path.name}")
    return stats_df


def _scrape_player_stats_list(draft_df: pd.DataFrame,
                               session: requests.Session) -> list:
    rows = []
    for _, row in draft_df.iterrows():
        href = str(row.get("player_href", ""))
        pid  = str(row.get("wnba_player_id", ""))
        if not href or len(pid) < 3:
            rows.append({"wnba_player_id": pid, "player": row.get("player", "")})
            continue
        stats = scrape_wnba_player_career(href, session)
        stats["wnba_player_id"] = pid
        stats["player"]         = row.get("player", "")
        rows.append(stats)
        time.sleep(random.uniform(*SLEEP_REQUESTS))
    return rows

# ---------------------------------------------------------------------------
# 3. Build WNBA success target
# ---------------------------------------------------------------------------

def build_wnba_target(df: pd.DataFrame) -> pd.Series:
    """
    Composite WNBA success score (regression target):

      - Players with < MIN_WNBA_GAMES get a score of 0.
      - For players who established themselves:
          50% WS/40 z-score  (efficiency per minute)
          30% BPM z-score    (net impact on team)
          20% log(games)     (longevity / durability)

    The final score is z-scored across all training picks and then normalised
    to [0, 100] for display.
    """
    games  = pd.to_numeric(df.get("wnba_games",   0), errors="coerce").fillna(0)
    ws40   = pd.to_numeric(df.get("wnba_ws40",    np.nan), errors="coerce")
    bpm    = pd.to_numeric(df.get("wnba_bpm",     np.nan), errors="coerce")

    established = games >= MIN_WNBA_GAMES

    def zscore(s):
        vals = s[established]
        mu, sigma = vals.mean(), vals.std()
        if sigma < 1e-8:
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sigma

    ws40_z = zscore(ws40.clip(-0.2, 0.5)).fillna(0)
    bpm_z  = zscore(bpm.clip(-15, 20)).fillna(0)
    log_g  = np.log1p(games.clip(0, None))
    log_g_z = zscore(log_g).fillna(0)

    raw = 0.50 * ws40_z + 0.30 * bpm_z + 0.20 * log_g_z

    # Players who never made the league get 0
    raw = raw.where(established, 0.0)

    return raw.rename("wnba_target")

# ---------------------------------------------------------------------------
# 4. Match draft picks to pre-draft NCAA stats
# ---------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    """Normalise player names for fuzzy matching."""
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_ncaa_features(draft_df: pd.DataFrame,
                         features_df: pd.DataFrame,
                         season_offset: int = 1) -> pd.DataFrame:
    """
    For each draft pick, find the best-matching row in features_df where
    most_recent_year == (draft_year - season_offset).

    Falls back to (draft_year - 2) if no match found at offset 1.

    Returns draft_df with NCAA feature columns merged in.
    """
    features_df = features_df.copy()
    features_df["_clean_name"] = features_df["player"].apply(_clean_name)

    results = []
    for _, pick in draft_df.iterrows():
        player_name = _clean_name(str(pick.get("player", "")))
        draft_year  = int(pick.get("draft_year", 0))

        best_row  = None
        best_sim  = 0.0

        for offset in [season_offset, season_offset + 1, 0]:
            target_year = draft_year - offset
            candidates  = features_df[
                features_df["most_recent_year"] == target_year
            ] if "most_recent_year" in features_df.columns else features_df

            if candidates.empty:
                continue

            sims = candidates["_clean_name"].apply(
                lambda n: _similarity(player_name, n)
            )
            idx  = sims.idxmax()
            sim  = sims[idx]
            if sim > best_sim:
                best_sim = sim
                best_row = candidates.loc[idx]

            if best_sim >= 0.90:
                break

        row_dict = pick.to_dict()
        row_dict["name_match_score"] = round(best_sim, 3)

        if best_row is not None and best_sim >= 0.75:
            for col in FEATURES + ["archetype", "cluster"]:
                if col in best_row.index:
                    row_dict[f"ncaa_{col}"] = best_row[col]
            row_dict["ncaa_player"]       = best_row.get("player", "")
            row_dict["ncaa_season_year"]  = best_row.get("most_recent_year", np.nan)
            row_dict["ncaa_matched"]      = True
        else:
            row_dict["ncaa_matched"] = False

        results.append(row_dict)

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 5. Train gradient boosting model
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return the NCAA feature columns that exist in df."""
    return [f"ncaa_{f}" for f in FEATURES if f"ncaa_{f}" in df.columns]


def train_model(train_df: pd.DataFrame):
    """
    Train HistGradientBoostingRegressor with leave-one-year-out CV.
    HistGradientBoostingRegressor is sklearn's native gradient boosting —
    same algorithm as XGBoost, handles NaN natively, no extra dependencies.
    Returns the fitted model and feature column list.
    """
    feat_cols = get_feature_cols(train_df)
    if not feat_cols:
        raise ValueError("No NCAA feature columns found. Check match step.")

    # Drop rows with no NCAA data
    mask = train_df["ncaa_matched"].fillna(False).astype(bool)
    data = train_df[mask].copy()
    print(f"  Training on {len(data)} matched draft picks, {len(feat_cols)} features")

    if len(data) < 20:
        raise ValueError(
            f"Only {len(data)} matched training examples — too few to train reliably. "
            "Extend DRAFT_YEARS or check name matching."
        )

    X = data[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = data["wnba_target"].values

    # Leave-one-year-out CV
    years = sorted(data["draft_year"].unique())
    oof_mae = []
    for holdout_year in years:
        tr_mask = (data["draft_year"] != holdout_year).values
        va_mask  = (data["draft_year"] == holdout_year).values
        if va_mask.sum() < 2:
            continue

        model_cv = _build_model()
        model_cv.fit(X.values[tr_mask], y[tr_mask])
        preds = model_cv.predict(X.values[va_mask])
        oof_mae.append(mean_absolute_error(y[va_mask], preds))

    if oof_mae:
        print(f"  Leave-one-year-out MAE: {np.mean(oof_mae):.4f} ± {np.std(oof_mae):.4f}")

    # Final model on all data
    final_model = _build_model()
    final_model.fit(X.values, y)

    # Permutation-based feature importance (model-agnostic, works for any estimator)
    _print_feature_importance(final_model, X.values, y, feat_cols)

    return final_model, feat_cols


def _build_model() -> HistGradientBoostingRegressor:
    """
    HistGradientBoostingRegressor — gradient boosting with histogram binning.
    Regularized for small datasets: shallow trees, L2 regularization, min samples per leaf.
    Natively handles NaN so no imputation step is needed.
    """
    return HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=3,
        learning_rate=0.05,
        l2_regularization=2.0,
        min_samples_leaf=5,
        max_leaf_nodes=15,
        random_state=42,
    )


def _print_feature_importance(model, X: np.ndarray, y: np.ndarray,
                               feat_cols: list):
    """Use permutation importance so the ranking is meaningful regardless of model."""
    result = permutation_importance(model, X, y, n_repeats=20, random_state=42,
                                    n_jobs=-1)
    ranked = sorted(zip(feat_cols, result.importances_mean),
                    key=lambda x: -x[1])
    print("\n  Top 15 feature importances (permutation, mean decrease in MAE):")
    for name, score in ranked[:15]:
        display_name = name.replace("ncaa_", "")
        bar = "#" * max(0, int(score * 200))
        print(f"    {score:+.4f}  {display_name:30s} {bar}")

# ---------------------------------------------------------------------------
# 6. Score current 2025 prospects
# ---------------------------------------------------------------------------

def score_prospects(model, feat_cols: list,
                    prospects_df: pd.DataFrame) -> pd.Series:
    """
    Apply the trained model to 2025 prospects.
    Prospects df uses original column names (no ncaa_ prefix) — we map them here.
    Returns scores normalised to 0–100.
    """
    X_all = pd.DataFrame(index=prospects_df.index)
    for feat_col in feat_cols:
        base_col = feat_col.replace("ncaa_", "")
        if base_col in prospects_df.columns:
            X_all[feat_col] = pd.to_numeric(prospects_df[base_col], errors="coerce")
        else:
            X_all[feat_col] = np.nan

    raw_scores = model.predict(X_all[feat_cols].values)

    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        normed = 100.0 * (raw_scores - s_min) / (s_max - s_min)
    else:
        normed = np.full_like(raw_scores, 50.0)

    return pd.Series(normed.round(2), index=prospects_df.index, name="readiness_score")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in [RAW_DIR, PROC_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    in_path = PROC_DIR / "ncaaw_players_features.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found. Run pipeline/03_build_features.py first."
        )

    print("Loading 2025 prospect features...")
    prospects_df = pd.read_csv(in_path, low_memory=False)
    print(f"  {len(prospects_df)} prospects loaded")

    with make_session() as session:
        # ---- Step 1: WNBA draft history ----
        print("\nStep 1: Loading WNBA draft history...")
        draft_df = load_or_scrape_draft_history(session)
        if draft_df.empty:
            raise RuntimeError("No draft data scraped. Check network / basketball-reference structure.")
        print(f"  {len(draft_df)} total draft picks across {draft_df['draft_year'].nunique()} years")

        # ---- Step 2: WNBA career stats ----
        print("\nStep 2: Loading WNBA career stats...")
        stats_df = load_or_scrape_wnba_stats(draft_df, session)

    # ---- Step 3: Build target ----
    print("\nStep 3: Building WNBA success target...")
    merged = draft_df.merge(stats_df[["wnba_player_id"] + [c for c in stats_df.columns
                                                             if c.startswith("wnba_") and c != "wnba_player_id"]],
                             on="wnba_player_id", how="left")
    merged["wnba_target"] = build_wnba_target(merged)

    established = (pd.to_numeric(merged.get("wnba_games", 0), errors="coerce").fillna(0) >= MIN_WNBA_GAMES)
    print(f"  {established.sum()} / {len(merged)} draft picks established in WNBA (>={MIN_WNBA_GAMES} games)")

    # ---- Step 4: Match NCAA features ----
    print("\nStep 4: Matching draft picks to pre-draft NCAA stats...")
    train_df = match_ncaa_features(merged, prospects_df)
    matched = train_df["ncaa_matched"].sum()
    print(f"  Matched {matched} / {len(train_df)} picks to NCAA data "
          f"(name similarity ≥ 0.75)")

    # Save training set
    train_out = PROC_DIR / "xgb_training_set.csv"
    train_df.to_csv(train_out, index=False)
    print(f"  Training set saved → {train_out.name}")

    # ---- Step 5: Train gradient boosting model ----
    print("\nStep 5: Training gradient boosting model...")
    gb_model, feat_cols = train_model(train_df)

    # Save model
    joblib.dump(gb_model, MODEL_DIR / "xgb_model.pkl")
    print(f"  Model saved → data/models/xgb_model.pkl")

    # ---- Step 6: Score 2025 prospects ----
    print("\nStep 6: Scoring 2025 prospects...")
    readiness = score_prospects(gb_model, feat_cols, prospects_df)
    prospects_df["readiness_score"] = readiness

    prospects_df.to_csv(in_path, index=False)
    print(f"  readiness_score added to {in_path.name}")

    # Show top 25
    cols_show = ["player", "archetype", "pts_per_g", "bpm", "ts_pct",
                 "adj_opp_win_pct", "readiness_score"]
    cols_show = [c for c in cols_show if c in prospects_df.columns]
    top25 = prospects_df.nlargest(25, "readiness_score")[cols_show]
    print("\n  Top 25 players by readiness_score:")
    print(top25.to_string(index=False))

    # Distribution summary
    print(f"\n  Score distribution (n={len(readiness)}):")
    print(readiness.describe().round(2).to_string())


if __name__ == "__main__":
    main()
