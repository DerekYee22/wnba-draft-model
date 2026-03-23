import { useState } from 'react'
import { ARCHETYPE_META, ARCHETYPES } from '../data/teams'
import { rankPlayersForTeam, fmt, pct } from '../utils/fitScore'

const POS_OPTIONS = ['All', 'G', 'F', 'C']

export default function PlayerTable({ players, needs, drafted, onDraft, showDrafted = false, teamId }) {
  const [archFilter, setArchFilter] = useState('All')
  const [posFilter, setPosFilter] = useState('All')
  const [showDraftedRows, setShowDraftedRows] = useState(false)
  const [expandedPlayer, setExpandedPlayer] = useState(null)

  // All players ranked for this team (excluding drafted unless showDraftedRows)
  const available = rankPlayersForTeam(players, needs, showDraftedRows ? new Set() : drafted, teamId)

  const filtered = available.filter((p) => {
    const archOk = archFilter === 'All' || p.archetype === archFilter
    const posOk = posFilter === 'All' || p.pos === posFilter
    return archOk && posOk
  })

  // Show drafted ones separately at bottom
  const draftedRows = showDraftedRows
    ? players
        .filter((p) => drafted.has(p.name + '|' + p.team))
        .map((p) => ({ ...p, fitScore: 0, isDrafted: true }))
    : []

  const rows = [...filtered.map((p) => ({ ...p, isDrafted: false })), ...draftedRows]

  return (
    <div className="space-y-3">
      {/* Controls */}
      <div className="flex flex-wrap gap-2 items-center">
        <div className="flex gap-1">
          <span className="text-xs text-gray-500 self-center mr-1">Archetype</span>
          {['All', ...ARCHETYPES].map((a) => {
            const meta = a !== 'All' ? ARCHETYPE_META[a] : null
            return (
              <button
                key={a}
                onClick={() => setArchFilter(a)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  archFilter === a
                    ? 'bg-gray-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                {a === 'All' ? 'All' : (
                  <span>{meta.icon} {a}</span>
                )}
              </button>
            )
          })}
        </div>
        <div className="flex gap-1">
          <span className="text-xs text-gray-500 self-center mr-1">Pos</span>
          {POS_OPTIONS.map((pos) => (
            <button
              key={pos}
              onClick={() => setPosFilter(pos)}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                posFilter === pos
                  ? 'bg-gray-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {pos}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer ml-auto">
          <input
            type="checkbox"
            checked={showDraftedRows}
            onChange={(e) => setShowDraftedRows(e.target.checked)}
            className="rounded"
          />
          Show drafted
        </label>
      </div>

      <div className="text-xs text-gray-500">
        Showing {filtered.length} available players
        {drafted.size > 0 && ` · ${drafted.size} drafted`}
      </div>

      {/* Table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-medium w-8">#</th>
                <th className="text-left px-4 py-3 font-medium">Player</th>
                <th className="text-left px-4 py-3 font-medium hidden md:table-cell">Archetype</th>
                <th className="text-center px-3 py-3 font-medium">Fit</th>
                <th className="text-right px-3 py-3 font-medium hidden sm:table-cell">PTS</th>
                <th className="text-right px-3 py-3 font-medium hidden sm:table-cell">REB</th>
                <th className="text-right px-3 py-3 font-medium hidden lg:table-cell">AST</th>
                <th className="text-right px-3 py-3 font-medium hidden lg:table-cell">STL</th>
                <th className="text-right px-3 py-3 font-medium hidden lg:table-cell">BLK</th>
                <th className="text-right px-3 py-3 font-medium hidden xl:table-cell">TS%</th>
                <th className="text-right px-3 py-3 font-medium hidden xl:table-cell">BPM</th>
                <th className="px-3 py-3 font-medium w-24"></th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={12} className="text-center py-12 text-gray-500">
                    No players match the selected filters.
                  </td>
                </tr>
              )}
              {rows.map((player, idx) => {
                const key = player.name + '|' + player.team
                const meta = ARCHETYPE_META[player.archetype] ?? {}
                const isExpanded = expandedPlayer === key
                const isDrafted = drafted.has(key)

                return [
                  <tr
                    key={key}
                    className={`border-b border-gray-800/50 hover:bg-gray-800/40 transition-colors cursor-pointer ${
                      isDrafted ? 'opacity-40' : ''
                    }`}
                    onClick={() => setExpandedPlayer(isExpanded ? null : key)}
                  >
                    <td className="px-4 py-3 text-gray-500 font-mono text-xs">
                      {isDrafted ? '—' : idx + 1}
                    </td>
                    <td className="px-4 py-3">
                      <div className="font-semibold text-white">{player.name}</div>
                      <div className="text-xs text-gray-400">
                        {player.pos} · {player.team} · {player.conference?.replace(' Conference', '')}
                      </div>
                    </td>
                    <td className="px-4 py-3 hidden md:table-cell">
                      <span
                        className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border ${meta.bg} ${meta.border} ${meta.text}`}
                      >
                        {meta.icon} {player.archetype}
                      </span>
                    </td>
                    <td className="px-3 py-3 text-center">
                      {!isDrafted ? (
                        <FitBadge score={player.fitScore} />
                      ) : (
                        <span className="text-xs text-gray-600">Drafted</span>
                      )}
                    </td>
                    <td className="px-3 py-3 text-right hidden sm:table-cell text-gray-300">
                      {fmt(player.pts_per_g)}
                    </td>
                    <td className="px-3 py-3 text-right hidden sm:table-cell text-gray-300">
                      {fmt(player.treb_per_g)}
                    </td>
                    <td className="px-3 py-3 text-right hidden lg:table-cell text-gray-300">
                      {fmt(player.ast_per_g)}
                    </td>
                    <td className="px-3 py-3 text-right hidden lg:table-cell text-gray-300">
                      {fmt(player.stl_per_g)}
                    </td>
                    <td className="px-3 py-3 text-right hidden lg:table-cell text-gray-300">
                      {fmt(player.blk_per_g)}
                    </td>
                    <td className="px-3 py-3 text-right hidden xl:table-cell text-gray-300">
                      {pct(player.ts_pct)}
                    </td>
                    <td className="px-3 py-3 text-right hidden xl:table-cell text-gray-300">
                      {fmt(player.bpm)}
                    </td>
                    <td className="px-3 py-3 text-right">
                      {!isDrafted ? (
                        <button
                          onClick={(e) => { e.stopPropagation(); onDraft(player) }}
                          className="px-2.5 py-1 bg-[#FF6B2B] hover:bg-orange-500 text-white text-xs font-medium rounded transition-colors whitespace-nowrap"
                        >
                          Draft
                        </button>
                      ) : (
                        <button
                          onClick={(e) => { e.stopPropagation(); /* handled in board */ }}
                          className="px-2.5 py-1 bg-gray-700 text-gray-400 text-xs font-medium rounded cursor-default"
                          disabled
                        >
                          Drafted
                        </button>
                      )}
                    </td>
                  </tr>,
                  isExpanded && (
                    <tr key={key + '_expanded'} className="bg-gray-800/30">
                      <td colSpan={12} className="px-6 py-4">
                        <PlayerDetail player={player} />
                      </td>
                    </tr>
                  ),
                ]
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function FitBadge({ score }) {
  const color =
    score >= 70 ? 'text-emerald-400 bg-emerald-500/20' :
    score >= 45 ? 'text-amber-400 bg-amber-500/20' :
    'text-gray-400 bg-gray-700'

  return (
    <span className={`inline-block text-xs font-bold px-2 py-0.5 rounded ${color}`}>
      {score.toFixed(0)}
    </span>
  )
}

function PlayerDetail({ player }) {
  const stats = [
    { label: 'MPG', value: fmt(player.mpg) },
    { label: 'Games', value: player.games_played },
    { label: 'PTS', value: fmt(player.pts_per_g) },
    { label: 'REB', value: fmt(player.treb_per_g) },
    { label: 'AST', value: fmt(player.ast_per_g) },
    { label: 'STL', value: fmt(player.stl_per_g) },
    { label: 'BLK', value: fmt(player.blk_per_g) },
    { label: 'TOV', value: fmt(player.tov_per_g) },
    { label: 'FG%', value: pct(player.fg_pct) },
    { label: '3P%', value: pct(player.fg3_pct) },
    { label: 'FT%', value: pct(player.ft_pct) },
    { label: 'TS%', value: pct(player.ts_pct) },
    { label: 'PER', value: fmt(player.per) },
    { label: 'BPM', value: fmt(player.bpm) },
    { label: 'WS/40', value: fmt(player.ws_per_40, 3) },
    { label: 'Readiness', value: player.readiness_score != null ? fmt(player.readiness_score, 1) : '—' },
    { label: 'Arch Score', value: fmt(player.archetype_score, 3) },
    { label: 'Arch Rank', value: player.rank_in_archetype ? `#${Math.round(player.rank_in_archetype)}` : '—' },
    { label: 'Team W-L', value: player.record ?? '—' },
  ]

  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <div>
          <span className="font-bold text-white">{player.name}</span>
          <span className="text-gray-400 text-sm ml-2">
            {player.pos} · {player.team} · {player.conference}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-3 sm:grid-cols-6 lg:grid-cols-9 gap-2">
        {stats.map(({ label, value }) => (
          <div key={label} className="bg-gray-900 rounded-lg p-2 text-center">
            <div className="text-xs text-gray-500 mb-0.5">{label}</div>
            <div className="text-sm font-semibold text-white">{value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
