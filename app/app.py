"""
WNBA Draft Board — Streamlit Application
==========================================
Interactive draft simulator that ranks NCAA players by their fit for the WNBA
team currently on the clock, updating in real-time as players are selected.

Run:
    streamlit run app/app.py

Data sources (in priority order):
    1. data/processed/player_fit_scores.csv  (full pipeline output)
    2. ncaaw_players_with_archetypes_ranked.csv  (fallback: existing single-year data)
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"

ARCHETYPE_COLORS = {
    "Primary Creator":      "#E63946",
    "Balanced Contributor": "#F4A261",
    "Interior Defender":    "#2A9D8F",
    "Support Player":       "#457B9D",
    "Unknown":              "#6C757D",
}

WNBA_TEAMS = [
    "Dallas Wings",
    "Chicago Sky",
    "Washington Mystics",
    "Los Angeles Sparks",
    "Golden State Valkyries",
    "Indiana Fever",
    "Seattle Storm",
    "Atlanta Dream",
    "Phoenix Mercury",
    "Connecticut Sun",
    "Las Vegas Aces",
    "New York Liberty",
    "Minnesota Lynx",
]

TEAM_SHORT = {
    "Dallas Wings":          "DAL",
    "Chicago Sky":           "CHI",
    "Washington Mystics":    "WAS",
    "Los Angeles Sparks":    "LAL",
    "Golden State Valkyries":"GSV",
    "Indiana Fever":         "IND",
    "Seattle Storm":         "SEA",
    "Atlanta Dream":         "ATL",
    "Phoenix Mercury":       "PHX",
    "Connecticut Sun":       "CON",
    "Las Vegas Aces":        "LVA",
    "New York Liberty":      "NYL",
    "Minnesota Lynx":        "MIN",
}

# Hardcoded team needs labels for the sidebar (top 3 per team)
TEAM_NEEDS_LABELS = {
    "Atlanta Dream":         ["Shooting Efficiency", "True Shooting", "2-Pt Creation"],
    "Chicago Sky":           ["True Shooting", "2-Pt Creation", "Scoring"],
    "Connecticut Sun":       ["True Shooting", "Defensive Rebounding", "2-Pt Creation"],
    "Dallas Wings":          ["Perimeter Defense", "Interior Defense", "Rebounding"],
    "Golden State Valkyries":["Scoring", "Playmaking", "Rebounding"],
    "Indiana Fever":         ["Perimeter Defense", "Interior Defense", "Playmaking"],
    "Las Vegas Aces":        ["3-Pt Defense", "Rebounding", "Playmaking"],
    "Los Angeles Sparks":    ["Perimeter Defense", "Scoring", "Interior Defense"],
    "Minnesota Lynx":        ["2-Pt Creation", "Depth", "Rebounding"],
    "New York Liberty":      ["Depth", "Versatility", "Wing Defense"],
    "Phoenix Mercury":       ["Playmaking Defense", "Rebounding", "Playmaking"],
    "Seattle Storm":         ["True Shooting", "Defensive Rebounding", "Playmaking"],
    "Washington Mystics":    ["Scoring", "Rebounding", "2-Pt Creation"],
}

# Default 2025 draft order (snake, 3 rounds)
def build_default_draft_order(teams=WNBA_TEAMS, n_rounds=3) -> list[dict]:
    order = []
    for rnd in range(1, n_rounds + 1):
        team_list = teams if rnd % 2 == 1 else list(reversed(teams))
        for pick_in_round, team in enumerate(team_list, 1):
            overall = (rnd - 1) * len(teams) + pick_in_round
            order.append({"round": rnd, "pick_in_round": pick_in_round,
                          "overall": overall, "team": team})
    return order

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_players() -> pd.DataFrame:
    """Load player data. Falls back to the existing archetype CSV if the pipeline
    hasn't been run yet."""
    full_path = PROC_DIR / "player_fit_scores.csv"
    if full_path.exists():
        df = pd.read_csv(full_path, low_memory=False)
        st.session_state["_data_source"] = "Full pipeline (multi-year + Ridge + fit scores)"
        return df

    # Fallback: existing single-year processed CSV
    fallback = ROOT / "ncaaw_players_with_archetypes_ranked.csv"
    if fallback.exists():
        df = pd.read_csv(fallback, low_memory=False)
        df = df.rename(columns={"name": "player"})
        # Build approximate fit scores from archetype rank as a proxy
        df = _build_fallback_fit_scores(df)
        st.session_state["_data_source"] = "Fallback: single-year archetype CSV (run pipeline for full scores)"
        return df

    st.error("No player data found. Run the pipeline or place ncaaw_players_with_archetypes_ranked.csv in the project root.")
    st.stop()


def _build_fallback_fit_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    When the full pipeline hasn't run, build approximate fit scores from
    archetype_score and basic stats already in the CSV.
    """
    # Use archetype_score as a proxy for readiness_score
    if "archetype_score" in df.columns:
        score = pd.to_numeric(df["archetype_score"], errors="coerce").fillna(0)
        s_min, s_max = score.min(), score.max()
        if s_max > s_min:
            df["readiness_score"] = 100 * (score - s_min) / (s_max - s_min)
        else:
            df["readiness_score"] = 50.0
    else:
        df["readiness_score"] = 50.0

    # Build very rough team fit scores based on archetype + team needs
    # Primary Creators → fit teams needing scoring (ATL, CHI, WAS, LAL)
    # Interior Defenders → fit teams needing defense (DAL, IND)
    # Support Players → fit teams needing playmaking (PHX, CON, SEA)
    ARCHETYPE_TEAM_AFFINITY = {
        "Primary Creator":      ["Atlanta Dream", "Chicago Sky", "Washington Mystics",
                                  "Los Angeles Sparks", "Dallas Wings"],
        "Balanced Contributor": ["Indiana Fever", "Connecticut Sun", "Seattle Storm",
                                  "Las Vegas Aces", "Golden State Valkyries"],
        "Interior Defender":    ["Dallas Wings", "Indiana Fever", "Phoenix Mercury",
                                  "Washington Mystics", "Los Angeles Sparks"],
        "Support Player":       ["Phoenix Mercury", "Connecticut Sun", "Seattle Storm",
                                  "Minnesota Lynx", "New York Liberty"],
    }

    rs_z = (df["readiness_score"] - df["readiness_score"].mean()) / (df["readiness_score"].std() + 1e-8)

    for team in WNBA_TEAMS:
        affinity = pd.Series(0.0, index=df.index)
        for arch, teams in ARCHETYPE_TEAM_AFFINITY.items():
            if team in teams:
                affinity += (df["archetype"] == arch).astype(float) * 0.5
        fit_z = affinity
        df[f"{team}_fit"]   = fit_z.round(4)
        df[f"{team}_total"] = (0.6 * rs_z + 0.4 * fit_z).round(4)

    return df


@st.cache_data
def load_team_needs() -> pd.DataFrame | None:
    path = PROC_DIR / "wnba_top5_needs.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_state(players: pd.DataFrame):
    if "draft_initialized" not in st.session_state:
        st.session_state.draft_initialized  = True
        st.session_state.pick_order         = build_default_draft_order()
        st.session_state.current_pick_idx   = 0
        st.session_state.drafted            = {}   # {team: [player_name, ...]}
        st.session_state.available          = set(players["player"].tolist())
        st.session_state.draft_log          = []
        st.session_state.readiness_weight   = 0.55
        st.session_state.show_archetype     = "All"
        st.session_state.show_position      = "All"
        st.session_state.min_readiness      = 0

# ---------------------------------------------------------------------------
# Draft logic
# ---------------------------------------------------------------------------

def draft_player(player_name: str, players: pd.DataFrame):
    if st.session_state.current_pick_idx >= len(st.session_state.pick_order):
        return  # draft over

    pick_info = st.session_state.pick_order[st.session_state.current_pick_idx]
    team      = pick_info["team"]

    if team not in st.session_state.drafted:
        st.session_state.drafted[team] = []
    st.session_state.drafted[team].append(player_name)
    st.session_state.available.discard(player_name)

    # Get archetype for log
    arch_row = players[players["player"] == player_name]
    archetype = arch_row["archetype"].iloc[0] if not arch_row.empty and "archetype" in arch_row.columns else "?"

    st.session_state.draft_log.append({
        "Pick":      pick_info["overall"],
        "Round":     pick_info["round"],
        "Team":      team,
        "Player":    player_name,
        "Archetype": archetype,
    })
    st.session_state.current_pick_idx += 1
    st.rerun()


def undo_last_pick():
    if not st.session_state.draft_log:
        return
    last = st.session_state.draft_log.pop()
    team   = last["Team"]
    player = last["Player"]
    st.session_state.drafted.get(team, []).remove(player) if player in st.session_state.drafted.get(team, []) else None
    st.session_state.available.add(player)
    st.session_state.current_pick_idx = max(0, st.session_state.current_pick_idx - 1)
    st.rerun()


def reset_draft():
    keys_to_clear = [k for k in st.session_state if k != "draft_initialized"]
    for k in keys_to_clear:
        del st.session_state[k]
    st.session_state.draft_initialized = False
    st.rerun()

# ---------------------------------------------------------------------------
# Rankings computation
# ---------------------------------------------------------------------------

def get_ranked_players(players: pd.DataFrame, team: str,
                        rw: float, arch_filter: str,
                        pos_filter: str, min_rs: float) -> pd.DataFrame:
    """Filter available players and rank by total score for the given team."""
    avail = players[players["player"].isin(st.session_state.available)].copy()

    # Re-compute total score with the current readiness weight slider
    total_col = f"{team}_total"
    fit_col   = f"{team}_fit"
    if total_col in avail.columns and fit_col in avail.columns and "readiness_score" in avail.columns:
        rs_z  = _zscore(avail["readiness_score"])
        fit_z = _zscore(avail[fit_col])
        avail["_total_live"] = (rw * rs_z + (1 - rw) * fit_z).round(3)
    elif total_col in avail.columns:
        avail["_total_live"] = avail[total_col]
    else:
        avail["_total_live"] = avail.get("readiness_score", pd.Series(50.0, index=avail.index))

    # Filters
    if arch_filter != "All" and "archetype" in avail.columns:
        avail = avail[avail["archetype"] == arch_filter]
    if pos_filter != "All" and "pos" in avail.columns:
        avail = avail[avail["pos"] == pos_filter]
    if min_rs > 0 and "readiness_score" in avail.columns:
        avail = avail[avail["readiness_score"] >= min_rs]

    return avail.sort_values("_total_live", ascending=False).reset_index(drop=True)


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def archetype_badge(archetype: str) -> str:
    color = ARCHETYPE_COLORS.get(archetype, ARCHETYPE_COLORS["Unknown"])
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600">{archetype}</span>'


def render_team_needs_bar(team: str):
    needs_df = load_team_needs()
    if needs_df is not None:
        team_needs = needs_df[needs_df["Team"] == team].head(5)
        if not team_needs.empty:
            fig = go.Figure(go.Bar(
                x=team_needs["weighted_deficit"].clip(lower=0),
                y=team_needs["stat"],
                orientation="h",
                marker_color="#457B9D",
                text=team_needs["weighted_deficit"].round(3).astype(str),
                textposition="outside",
            ))
            fig.update_layout(
                height=180, margin=dict(l=10, r=30, t=10, b=10),
                xaxis_title="Deficit", yaxis_title="",
                font=dict(size=11),
            )
            st.plotly_chart(fig, use_container_width=True)
            return

    # Fallback: show text labels
    labels = TEAM_NEEDS_LABELS.get(team, [])
    if labels:
        st.write(" · ".join(f"**{l}**" for l in labels))


def render_draft_board_table(ranked: pd.DataFrame, team: str, players_full: pd.DataFrame):
    display_cols_map = {
        "player":          "Player",
        "pos":             "Pos",
        "team":            "School",
        "conference":      "Conf",
        "archetype":       "Archetype",
        "readiness_score": "Readiness",
        "_total_live":     "Total Score",
        "pts_per_g":       "PPG",
        "ast_per_g":       "APG",
        "treb_per_g":      "RPG",
        "ts_pct":          "TS%",
        "bpm":             "BPM",
    }
    cols_present = {k: v for k, v in display_cols_map.items() if k in ranked.columns}
    display_df   = ranked[list(cols_present.keys())].rename(columns=cols_present).head(75)

    # Format numbers
    for col in ["Readiness", "Total Score", "PPG", "APG", "RPG", "BPM"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)
    if "TS%" in display_df.columns:
        display_df["TS%"] = pd.to_numeric(display_df["TS%"], errors="coerce").round(3)

    display_df.insert(0, "Rank", range(1, len(display_df) + 1))

    # Show table
    st.dataframe(display_df, use_container_width=True, height=480,
                 hide_index=True)

    # Draft button below the table
    st.write("---")
    pick_cols = st.columns([3, 1])
    with pick_cols[0]:
        available_names = ranked["player"].tolist()
        selected = st.selectbox(
            f"Select player for {team}",
            options=available_names,
            key=f"pick_select_{st.session_state.current_pick_idx}",
        )
    with pick_cols[1]:
        st.write("")
        st.write("")
        if st.button(f"Draft {selected}", type="primary", use_container_width=True):
            draft_player(selected, players_full)

# ---------------------------------------------------------------------------
# Save / load draft state
# ---------------------------------------------------------------------------

def save_draft():
    state_to_save = {
        "pick_order":       st.session_state.pick_order,
        "current_pick_idx": st.session_state.current_pick_idx,
        "drafted":          {k: list(v) for k, v in st.session_state.drafted.items()},
        "available":        list(st.session_state.available),
        "draft_log":        st.session_state.draft_log,
    }
    save_path = ROOT / "data" / "draft_autosave.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(state_to_save, indent=2))
    st.success(f"Draft saved to {save_path.name}")


def load_draft(players: pd.DataFrame):
    save_path = ROOT / "data" / "draft_autosave.json"
    if not save_path.exists():
        st.warning("No saved draft found.")
        return
    data = json.loads(save_path.read_text())
    st.session_state.pick_order       = data["pick_order"]
    st.session_state.current_pick_idx = data["current_pick_idx"]
    st.session_state.drafted          = {k: list(v) for k, v in data["drafted"].items()}
    st.session_state.available        = set(data["available"])
    st.session_state.draft_log        = data["draft_log"]
    st.success("Draft loaded.")
    st.rerun()

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="WNBA Draft Board",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
    .pick-header { font-size: 1.5rem; font-weight: 700; }
    .team-on-clock { font-size: 2rem; font-weight: 800; color: #E63946; }
    .needs-label { font-size: 0.8rem; color: #666; margin-bottom: 4px; }
    </style>
    """, unsafe_allow_html=True)

    # Load data
    players = load_players()
    init_state(players)

    # ---- SIDEBAR ----
    with st.sidebar:
        st.title("WNBA Draft Board")
        data_src = st.session_state.get("_data_source", "")
        if data_src:
            st.caption(f"Data: {data_src}")

        st.divider()
        st.subheader("Draft Controls")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Undo", use_container_width=True):
                undo_last_pick()
        with col2:
            if st.button("Save", use_container_width=True):
                save_draft()
        with col3:
            if st.button("Load", use_container_width=True):
                load_draft(players)

        if st.button("Reset Draft", type="secondary", use_container_width=True):
            reset_draft()

        st.divider()
        st.subheader("Model Weights")
        st.session_state.readiness_weight = st.slider(
            "Readiness vs. Fit",
            min_value=0.0, max_value=1.0,
            value=st.session_state.get("readiness_weight", 0.55),
            step=0.05,
            help="Higher → rank by global WNBA readiness. Lower → rank by team-specific fit.",
        )
        rw_label = f"**{st.session_state.readiness_weight:.0%} Readiness** / {1-st.session_state.readiness_weight:.0%} Team Fit"
        st.caption(rw_label)

        st.divider()
        st.subheader("Filters")
        arch_options = ["All"] + sorted(
            [a for a in players["archetype"].dropna().unique() if isinstance(a, str)]
        ) if "archetype" in players.columns else ["All"]
        st.session_state.show_archetype = st.selectbox("Archetype", arch_options)

        pos_options = ["All"] + sorted(
            [p for p in players["pos"].dropna().unique() if isinstance(p, str)]
        ) if "pos" in players.columns else ["All"]
        st.session_state.show_position = st.selectbox("Position", pos_options)

        st.session_state.min_readiness = st.slider(
            "Min Readiness Score", 0, 100,
            value=st.session_state.get("min_readiness", 0),
        )

        st.divider()
        st.subheader("Draft Progress")
        total_picks  = len(st.session_state.pick_order)
        current_pick = st.session_state.current_pick_idx
        st.progress(current_pick / total_picks if total_picks else 0)
        st.caption(f"Pick {current_pick} / {total_picks}")

        # Team rosters
        st.divider()
        st.subheader("Team Rosters")
        for team in WNBA_TEAMS:
            roster = st.session_state.drafted.get(team, [])
            short  = TEAM_SHORT.get(team, team)
            with st.expander(f"{short} ({len(roster)} picks)"):
                if roster:
                    for i, p in enumerate(roster, 1):
                        st.write(f"{i}. {p}")
                else:
                    st.caption("No picks yet")

    # ---- MAIN PANEL ----
    is_draft_over = st.session_state.current_pick_idx >= len(st.session_state.pick_order)

    if is_draft_over:
        st.markdown('<p class="team-on-clock">Draft Complete!</p>', unsafe_allow_html=True)
        st.balloons()
    else:
        pick_info      = st.session_state.pick_order[st.session_state.current_pick_idx]
        current_team   = pick_info["team"]
        current_overall = pick_info["overall"]
        current_round  = pick_info["round"]
        picks_remaining = len(st.session_state.pick_order) - st.session_state.current_pick_idx

        # Header row
        hcol1, hcol2, hcol3 = st.columns([3, 2, 1])
        with hcol1:
            st.markdown(
                f'<p class="pick-header">Pick #{current_overall} · Round {current_round}</p>'
                f'<p class="team-on-clock">{current_team}</p>',
                unsafe_allow_html=True,
            )
        with hcol2:
            st.markdown('<p class="needs-label">TOP NEEDS</p>', unsafe_allow_html=True)
            render_team_needs_bar(current_team)
        with hcol3:
            st.metric("Picks Remaining", picks_remaining)
            st.metric("Available Players",
                      len(players[players["player"].isin(st.session_state.available)]))

        st.divider()

        # Tabs
        tab_board, tab_team, tab_log = st.tabs(["Draft Board", "Team View", "Draft Log"])

        with tab_board:
            ranked = get_ranked_players(
                players, current_team,
                st.session_state.readiness_weight,
                st.session_state.show_archetype,
                st.session_state.show_position,
                st.session_state.min_readiness,
            )
            if ranked.empty:
                st.warning("No players match the current filters.")
            else:
                render_draft_board_table(ranked, current_team, players)

        with tab_team:
            st.subheader("View Board for Any Team")
            view_team = st.selectbox("Select team", WNBA_TEAMS,
                                     index=WNBA_TEAMS.index(current_team),
                                     key="team_view_select")
            ranked_view = get_ranked_players(
                players, view_team,
                st.session_state.readiness_weight,
                st.session_state.show_archetype,
                st.session_state.show_position,
                st.session_state.min_readiness,
            )
            if not ranked_view.empty:
                st.caption(f"Top 20 fits for **{view_team}**")
                render_draft_board_table(ranked_view.head(20), view_team, players)

            st.subheader(f"{view_team} Team Needs")
            render_team_needs_bar(view_team)

        with tab_log:
            st.subheader("Draft History")
            if st.session_state.draft_log:
                log_df = pd.DataFrame(st.session_state.draft_log)
                st.dataframe(log_df, use_container_width=True, hide_index=True)
            else:
                st.info("No picks made yet.")


if __name__ == "__main__":
    main()
