// Calculate how well a player fits a team's needs.
// Returns a score from 0–100.
//
// Priority order:
//   1. Pre-computed fit score embedded by prepare_data.py  (most accurate — full Ridge + stat model)
//   2. Archetype-based fallback                            (used when pipeline hasn't run)
export function calcFitScore(player, teamNeeds, teamId) {
  // 1. Pre-computed score from the modeling pipeline
  if (teamId && player.fitScores && player.fitScores[teamId] != null) {
    return player.fitScores[teamId]
  }

  // 2. Archetype-based fallback (original formula + reliability penalty)
  if (!teamNeeds || !player.archetype) return 0

  const needScore = teamNeeds.archetypeNeeds?.[player.archetype] ?? 5 // 0–10
  const archetypeScore = player.archetype_score ?? 1
  const rank = player.rank_in_archetype ?? 999

  // Quality factor: higher archetype score, lower rank number = better
  const qualityFactor = archetypeScore / Math.sqrt(rank)

  const raw = (needScore / 10) * qualityFactor * 100

  // Reliability: penalize players with low minutes. Advanced stats (BPM, TS%) are
  // very noisy for players with 10-14 MPG — a player needs real volume to trust.
  // 20 MPG = full credit, 15 MPG = 75%, 10 MPG = 50%
  const mpg = player.mpg ?? 0
  const games = player.games_played ?? 0
  const reliability = Math.min(1, mpg / 20) * Math.min(1, games / 20)

  return Math.min(100, Math.round(raw * reliability * 10) / 10)
}

// Sort players by fit score for a given team, filtered by availability
export function rankPlayersForTeam(players, teamNeeds, draftedSet, teamId) {
  return players
    .filter((p) => !draftedSet.has(p.name + '|' + p.team))
    .map((p) => ({ ...p, fitScore: calcFitScore(p, teamNeeds, teamId) }))
    .sort((a, b) => b.fitScore - a.fitScore)
}

// Stat display helpers
export function fmt(val, decimals = 1) {
  if (val == null || val === '' || isNaN(val)) return '—'
  return Number(val).toFixed(decimals)
}

export function pct(val) {
  if (val == null || val === '' || isNaN(val)) return '—'
  return (Number(val) * 100).toFixed(1) + '%'
}
