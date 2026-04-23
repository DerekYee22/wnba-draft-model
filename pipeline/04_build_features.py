"""
Feature Engineering — Multi-Year Aggregation
==============================================
Loads raw per-year player CSVs, applies exponential recency + minutes weighting,
merges measurables, and computes derived features.

Inputs:
    data/raw/ncaaw_players_raw_{year}.csv    (from 01_scrape_multi_year.py)
    data/raw/measurables_raw.csv             (from 02_scrape_measurables.py, optional)
    ncaaw_players_with_archetypes_ranked.csv (for archetype labels, fallback)

Outputs:
    data/processed/ncaaw_players_multiyear.csv  (long format, all player-seasons)
    data/processed/ncaaw_players_features.csv   (one row per player, aggregated)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent.parent
RAW_DIR      = ROOT / "data" / "raw"
PROC_DIR     = ROOT / "data" / "processed"
YEARS        = [2022, 2023, 2024, 2025, 2026]
DRAFT_YEAR   = 2026
EXISTING_CSV = ROOT / "ncaaw_players_with_archetypes_ranked.csv"

# Exponential recency decay: weight(y) = exp(LAMBDA * (y - BASE_YEAR))
BASE_YEAR = 2022
LAMBDA    = 0.5

# Minimum minutes per season to include a row
MIN_MP_SEASON = 60

# Minimum total career minutes to include a player in the aggregated output
MIN_MP_CAREER = 150

# Stats to aggregate (weighted average)
# "rate" stats = per-game or per-possession; averaged weighted by minutes
# "counting" stats = totals; summed, then re-derived per game
RATE_STATS = [
    "mpg", "fg_per_g", "fga_per_g", "fg_pct", "fg3_per_g", "fg3a_per_g", "fg3_pct",
    "fg2_per_g", "fg2a_per_g", "fg2_pct", "efg_pct", "ft_per_g", "fta_per_g", "ft_pct",
    "oreb_per_g", "dreb_per_g", "treb_per_g", "ast_per_g", "stl_per_g", "blk_per_g",
    "tov_per_g", "pf_per_g", "pts_per_g",
    "per", "ts_pct", "fg3a_per_fga_pct", "fta_per_fga_pct",
    "oreb_pct", "dreb_pct", "treb_pct", "ast_pct", "stl_pct", "blk_pct", "tov_pct",
    "usg_pct", "ows", "dws", "ws", "ws_per_40", "obpm", "dbpm", "bpm",
    "wins", "losses",
]

# These context columns are taken from the most recent season row
CONTEXT_COLS = ["conference", "team", "team_id", "player_href"]

# opp_win_pct is averaged across all available seasons (weighted by recency × minutes)
# rather than taken from only the most recent season, so players whose latest season
# lacks opp strength data (e.g. 2026 not yet scraped) still get a meaningful value.
OPP_WIN_PCT_COL = "opp_win_pct"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def year_weight(year: int) -> float:
    return float(np.exp(LAMBDA * (year - BASE_YEAR)))


def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def mpg_col(df: pd.DataFrame) -> pd.Series:
    """Return minutes per game, coercing to float."""
    for c in ("mpg", "mp_per_g", "mp"):
        if c in df.columns:
            return safe_float(df[c])
    return pd.Series(np.nan, index=df.index)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip pg_/adv_ prefixes that the scraper may have left, preferring pg_ on conflict.
    Also rename Sports-Reference stat names to the clean names used throughout the project.
    """
    SR_RENAME = {
        # per-game table
        "g":          "games_played",
        "gs":         "games_started",
        "mp_per_g":   "mpg",
        "fg_per_g":   "fg_per_g",
        "fga_per_g":  "fga_per_g",
        "fg_pct":     "fg_pct",
        "fg3_per_g":  "fg3_per_g",
        "fg3a_per_g": "fg3a_per_g",
        "fg3_pct":    "fg3_pct",
        "fg2_per_g":  "fg2_per_g",
        "fg2a_per_g": "fg2a_per_g",
        "fg2_pct":    "fg2_pct",
        "efg_pct":    "efg_pct",
        "ft_per_g":   "ft_per_g",
        "fta_per_g":  "fta_per_g",
        "ft_pct":     "ft_pct",
        "orb_per_g":  "oreb_per_g",
        "drb_per_g":  "dreb_per_g",
        "trb_per_g":  "treb_per_g",
        "ast_per_g":  "ast_per_g",
        "stl_per_g":  "stl_per_g",
        "blk_per_g":  "blk_per_g",
        "tov_per_g":  "tov_per_g",
        "pf_per_g":   "pf_per_g",
        "pts_per_g":  "pts_per_g",
        # advanced table
        "mp":         "mp",
        "orb_pct":    "oreb_pct",
        "drb_pct":    "dreb_pct",
        "trb_pct":    "treb_pct",
    }
    # First strip pg_/adv_ prefixes (prefer pg_ when both exist)
    pg_cols  = {c[3:]: c for c in df.columns if c.startswith("pg_")}
    adv_cols = {c[4:]: c for c in df.columns if c.startswith("adv_")}
    rename   = {}
    for base, pg_col_ in pg_cols.items():
        rename[pg_col_] = base
    for base, adv_col_ in adv_cols.items():
        if base not in pg_cols:
            rename[adv_col_] = base
    df = df.rename(columns=rename)

    # Then apply SR canonical name fixes
    df = df.rename(columns={k: v for k, v in SR_RENAME.items() if k in df.columns})
    return df

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

def load_raw_years(years) -> pd.DataFrame:
    frames = []
    for year in years:
        path = RAW_DIR / f"ncaaw_players_raw_{year}.csv"
        if not path.exists():
            print(f"  [WARN] {path.name} not found, skipping year {year}")
            continue
        df = pd.read_csv(path, low_memory=False)
        df = normalize_columns(df)
        df["season_year"] = year
        frames.append(df)

    if not frames:
        # Fall back to the existing processed CSV
        if EXISTING_CSV.exists():
            print("  [FALLBACK] Using existing ncaaw_players_with_archetypes_ranked.csv")
            df = pd.read_csv(EXISTING_CSV, low_memory=False)
            # Normalize: existing CSV uses "name", pipeline expects "player"
            if "name" in df.columns and "player" not in df.columns:
                df = df.rename(columns={"name": "player"})
            if "season_year" not in df.columns:
                df["season_year"] = 2025
            return df
        raise FileNotFoundError(
            "No raw player CSVs found and no existing CSV fallback. "
            "Run pipeline/01_scrape_multi_year.py first."
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(combined)} player-season rows across {len(frames)} years")
    return combined


# ---------------------------------------------------------------------------
# Deduplicate transfers (keep highest MPG row per player-year)
# ---------------------------------------------------------------------------

def dedup_transfers(df: pd.DataFrame) -> pd.DataFrame:
    df["_mpg"] = mpg_col(df)
    # Use player_id only when the column exists AND has non-null values
    id_col = (
        "player_id"
        if "player_id" in df.columns and df["player_id"].notna().any()
        else "player"
    )
    if "season_year" in df.columns:
        df = (
            df.sort_values("_mpg", ascending=False, na_position="last")
              .drop_duplicates(subset=[id_col, "season_year"], keep="first")
              .reset_index(drop=True)
        )
    df = df.drop(columns=["_mpg"], errors="ignore")
    return df

# ---------------------------------------------------------------------------
# Multi-year weighted aggregation
# ---------------------------------------------------------------------------

def aggregate_multi_year(long_df: pd.DataFrame) -> pd.DataFrame:
    """Produce one row per player with exponential recency × minutes weighting."""

    # Ensure all rate stats are numeric
    for col in RATE_STATS:
        if col in long_df.columns:
            long_df[col] = safe_float(long_df[col])

    # Minutes per season
    long_df["_mpg"]  = mpg_col(long_df)
    long_df["_mp"]   = safe_float(long_df.get("mp", long_df["_mpg"] * long_df.get("games_played", 1)))
    long_df["_yw"]   = long_df["season_year"].apply(year_weight)
    # Combined weight = recency × minutes (prevents low-minute seasons from polluting)
    long_df["_w"]    = long_df["_yw"] * long_df["_mp"].clip(lower=0).fillna(0)

    # Only keep rows meeting the minimum minutes threshold
    long_df = long_df[long_df["_mp"] >= MIN_MP_SEASON].copy()

    id_col = (
        "player_id"
        if "player_id" in long_df.columns and long_df["player_id"].notna().any()
        else "player"
    )
    rows = []
    for player_id, grp in long_df.groupby(id_col, sort=False):
        if len(grp) == 0:
            continue

        total_mp = grp["_mp"].sum()
        if total_mp < MIN_MP_CAREER:
            continue

        # Sort by year to get most-recent context
        grp_sorted = grp.sort_values("season_year", ascending=True)
        most_recent = grp_sorted.iloc[-1]

        # Resolve the canonical player name (column may be "player" or "name")
        player_name = most_recent.get("player") or most_recent.get("name") or str(player_id)

        # games_played in the most recent season (from 'g' or 'games_played' column)
        gp_latest = most_recent.get("games_played") or most_recent.get("g")
        try:
            gp_latest = int(float(gp_latest))
        except (TypeError, ValueError):
            gp_latest = None

        row = {
            "player_id":        player_id,
            "player":           player_name,
            "pos":              most_recent.get("pos", ""),
            "n_seasons":        len(grp),
            "first_season":     int(grp["season_year"].min()),
            "most_recent_year": int(grp["season_year"].max()),
            "total_mp":         total_mp,
            "mpg_latest":       float(most_recent.get("_mpg") or np.nan),
            "games_played_latest": gp_latest,
        }

        # Weighted average for each rate stat
        weights = grp["_w"].values
        total_w = weights.sum()
        for col in RATE_STATS:
            if col not in grp.columns:
                row[col] = np.nan
                continue
            vals = grp[col].values.astype(float)
            mask = ~np.isnan(vals)
            if mask.sum() == 0 or weights[mask].sum() == 0:
                row[col] = np.nan
            else:
                row[col] = float(np.average(vals[mask], weights=weights[mask]))

        # Latest-season values for display in the webapp (not used by the model)
        for col in RATE_STATS:
            latest_val = most_recent.get(col)
            try:
                row[f"{col}_latest"] = float(latest_val) if latest_val is not None else np.nan
            except (TypeError, ValueError):
                row[f"{col}_latest"] = np.nan

        # Season trend (slope of bpm and pts_per_g over time — reward upward trajectory)
        for trend_stat in ("bpm", "pts_per_g", "ws_per_40"):
            if trend_stat in grp.columns and grp["season_year"].nunique() >= 2:
                years_arr = grp["season_year"].values.astype(float)
                vals_arr  = safe_float(grp[trend_stat]).values
                mask      = ~np.isnan(vals_arr)
                if mask.sum() >= 2:
                    slope = float(np.polyfit(years_arr[mask], vals_arr[mask], 1)[0])
                else:
                    slope = 0.0
            else:
                slope = 0.0
            row[f"trend_{trend_stat}"] = slope

        # opp_win_pct: weighted average across seasons that have data.
        # Using all seasons (not just the most recent) prevents a missing
        # current-year lookup from zeroing out a player's SOS signal.
        if OPP_WIN_PCT_COL in grp.columns:
            opp_vals = safe_float(grp[OPP_WIN_PCT_COL]).values
            opp_mask = ~np.isnan(opp_vals)
            if opp_mask.sum() > 0 and weights[opp_mask].sum() > 0:
                row[OPP_WIN_PCT_COL] = float(
                    np.average(opp_vals[opp_mask], weights=weights[opp_mask])
                )
            else:
                row[OPP_WIN_PCT_COL] = np.nan
        else:
            row[OPP_WIN_PCT_COL] = np.nan

        # Context columns from most recent season
        for col in CONTEXT_COLS:
            row[col] = most_recent.get(col, np.nan)

        rows.append(row)

    agg_df = pd.DataFrame(rows)
    print(f"  Aggregated to {len(agg_df)} unique players")
    return agg_df

# ---------------------------------------------------------------------------
# Merge measurables
# ---------------------------------------------------------------------------

def merge_measurables(df: pd.DataFrame) -> pd.DataFrame:
    meas_path = RAW_DIR / "measurables_raw.csv"
    if not meas_path.exists():
        print("  [INFO] No measurables file found; height/weight/birth_year will be NaN")
        df["height_in"]  = np.nan
        df["weight_lbs"] = np.nan
        df["birth_year"] = np.nan
        return df

    keep_cols = ["player_id", "height_in", "weight_lbs"]
    meas_df = pd.read_csv(meas_path)
    if "birth_year" in meas_df.columns:
        keep_cols.append("birth_year")
    meas = meas_df[keep_cols].drop_duplicates("player_id")
    df   = df.merge(meas, on="player_id", how="left")
    if "birth_year" not in df.columns:
        df["birth_year"] = np.nan
    filled = df[["height_in", "weight_lbs"]].notna().sum()
    print(f"  Measurables merged: height={filled['height_in']}, weight={filled['weight_lbs']} / {len(df)}")
    return df

# ---------------------------------------------------------------------------
# Merge archetype labels from the existing processed CSV
# ---------------------------------------------------------------------------

def merge_archetype_labels(df: pd.DataFrame) -> pd.DataFrame:
    if not EXISTING_CSV.exists():
        print("  [INFO] No existing archetype CSV; cluster/archetype columns will be NaN")
        for col in ("cluster", "archetype", "archetype_score", "rank_in_archetype", "pca1", "pca2"):
            df[col] = np.nan
        return df

    arch = pd.read_csv(EXISTING_CSV, usecols=[
        "name", "cluster", "archetype", "archetype_score",
        "rank_in_archetype", "pca1", "pca2",
    ])
    arch = arch.rename(columns={"name": "player"})
    arch = arch.drop_duplicates("player")

    # Join on player name (best available key without player_id in the archetype CSV)
    df = df.merge(arch, on="player", how="left")
    filled = df["archetype"].notna().sum()
    print(f"  Archetype labels merged: {filled}/{len(df)} players matched")
    return df

# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # Assist-to-turnover ratio (safe against div/0)
    if "ast_per_g" in df.columns and "tov_per_g" in df.columns:
        df["ast_tov_ratio"] = df["ast_per_g"] / (df["tov_per_g"] + 0.1)

    # 3-point rate (share of attempts that are 3s)
    if "fg3a_per_g" in df.columns and "fga_per_g" in df.columns:
        df["three_rate"] = df["fg3a_per_g"] / (df["fga_per_g"] + 0.1)

    # Interior scoring proxy: 2pt makes weighted by efficiency
    if "fg2_per_g" in df.columns and "fg2_pct" in df.columns:
        df["inside_score"] = df["fg2_per_g"] * df["fg2_pct"].fillna(0)

    # Defensive composite proxy (normalised later in the Ridge model)
    if "blk_per_g" in df.columns and "stl_per_g" in df.columns and "dws" in df.columns:
        df["def_composite"] = df["blk_per_g"] + df["stl_per_g"] + df["dws"].fillna(0)

    # Boolean flag: multi-year player
    df["is_multi_year"] = (df.get("n_seasons", 1) > 1).astype(int)

    return df

# ---------------------------------------------------------------------------
# Fix first_season for transfers whose pre-transfer school wasn't scraped
# ---------------------------------------------------------------------------

def fix_transfer_first_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply true first_season values from the cache written by 01_scrape_multi_year.py.
    No HTTP requests — just a CSV lookup.
    """
    cache_path = RAW_DIR / "player_first_seasons.csv"
    if not cache_path.exists():
        print("  [INFO] player_first_seasons.csv not found — run 01_scrape_multi_year.py to build it")
        return df

    cache = pd.read_csv(cache_path)
    lookup = dict(zip(cache["player_id"].astype(str), cache["true_first_season"].astype(int)))

    updated = 0
    for idx, row in df.iterrows():
        pid = str(row.get("player_id", ""))
        if pid in lookup:
            true_first = lookup[pid]
            if true_first < df.at[idx, "first_season"]:
                print(f"    {row['player']}: first_season {df.at[idx, 'first_season']} → {true_first}")
                df.at[idx, "first_season"] = true_first
                updated += 1

    print(f"  Updated first_season for {updated} transfer players")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw player data...")
    long_df = load_raw_years(YEARS)

    # Merge opponent strength onto each player-season row.
    # When a team's opp_win_pct is missing for the current season (e.g. 2026 data
    # not yet scraped), fall back to the most recent prior year's value for that
    # team so freshmen and single-year players still get a meaningful SOS signal.
    opp_path = PROC_DIR / "team_opp_strength.csv"
    if opp_path.exists():
        print("Merging opponent strength...")
        opp_df = pd.read_csv(opp_path)[["season_year", "team_id", "opp_win_pct"]]
        long_df = long_df.merge(opp_df, on=["season_year", "team_id"], how="left")

        # Fallback: for rows still missing opp_win_pct, use the most recent
        # available value for that team from any earlier year.
        if long_df["opp_win_pct"].isna().any():
            latest_opp = (
                opp_df.sort_values("season_year")
                      .groupby("team_id")["opp_win_pct"]
                      .last()
                      .rename("opp_win_pct_fallback")
            )
            long_df = long_df.join(latest_opp, on="team_id")
            missing_mask = long_df["opp_win_pct"].isna()
            long_df.loc[missing_mask, "opp_win_pct"] = long_df.loc[missing_mask, "opp_win_pct_fallback"]
            long_df = long_df.drop(columns=["opp_win_pct_fallback"])
            fallback_filled = missing_mask.sum() - long_df["opp_win_pct"].isna().sum()
            print(f"  opp_win_pct fallback applied to {fallback_filled} rows (prior-year team value)")

        filled = long_df["opp_win_pct"].notna().sum()
        print(f"  opp_win_pct filled for {filled}/{len(long_df)} player-season rows")
    else:
        print("  [INFO] team_opp_strength.csv not found — run 03_opp_strength.py first")
        long_df["opp_win_pct"] = float("nan")

    # Save long-format (all seasons)
    long_out = PROC_DIR / "ncaaw_players_multiyear.csv"
    long_df.to_csv(long_out, index=False)
    print(f"  Long-format saved: {long_out.name} ({len(long_df)} rows)")

    # Deduplicate transfers
    print("Deduplicating transfers...")
    long_df = dedup_transfers(long_df)

    # Aggregate to one row per player
    print("Aggregating multi-year stats...")
    feat_df = aggregate_multi_year(long_df)

    # Fix first_season for transfers from unscraped conferences
    print("Resolving transfer first seasons...")
    feat_df = fix_transfer_first_seasons(feat_df)

    # Merge measurables
    print("Merging measurables...")
    feat_df = merge_measurables(feat_df)

    # Derived features
    feat_df = add_derived_features(feat_df)

    feat_out = PROC_DIR / "ncaaw_players_features.csv"
    feat_df.to_csv(feat_out, index=False)
    print(f"\nFeature CSV saved: {feat_out.name} ({len(feat_df)} players, {feat_df.shape[1]} columns)")
    print(f"  Run 05_archetype_clusters.py next to assign archetype labels.")
    print(feat_df[["player", "n_seasons", "pts_per_g", "bpm"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
