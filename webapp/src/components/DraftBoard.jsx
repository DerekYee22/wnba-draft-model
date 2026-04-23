import { ARCHETYPE_META } from '../data/teams'
import { fmt, pct } from '../utils/fitScore'

export default function DraftBoard({ players, drafted, draftLog = [], teams = [], onUndraft, onReset }) {
  const draftedPlayers = players.filter((p) => drafted.has(p.name + '|' + p.team))

  // Build a lookup from player key → draft log entry (for mock draft team display)
  const logByPlayerKey = {}
  draftLog.forEach((entry) => {
    if (entry) logByPlayerKey[entry.player.name + '|' + entry.player.team] = entry
  })

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Draft Board</h1>
          <p className="text-gray-400 text-sm mt-1">
            {draftedPlayers.length} player{draftedPlayers.length !== 1 ? 's' : ''} drafted
          </p>
        </div>
        {draftedPlayers.length > 0 && (
          <button
            onClick={() => { if (window.confirm('Reset the entire draft board?')) onReset() }}
            className="px-4 py-2 bg-red-900/40 border border-red-800 hover:bg-red-900/60 text-red-400 text-sm font-medium rounded-lg transition-colors"
          >
            Reset Draft
          </button>
        )}
      </div>

      {draftedPlayers.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
          <p className="text-gray-400 font-medium mb-2">No players drafted yet</p>
          <p className="text-gray-600 text-sm">
            Go to a team's page and click "Draft" next to a player to add them here.
          </p>
        </div>
      ) : (
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-medium">Pick</th>
                <th className="text-left px-4 py-3 font-medium">Player</th>
                {draftLog.some(Boolean) && (
                  <th className="text-left px-4 py-3 font-medium hidden sm:table-cell">Drafted By</th>
                )}
                <th className="text-left px-4 py-3 font-medium hidden md:table-cell">Archetype</th>
                <th className="text-right px-3 py-3 font-medium hidden sm:table-cell">PTS</th>
                <th className="text-right px-3 py-3 font-medium hidden sm:table-cell">REB</th>
                <th className="text-right px-3 py-3 font-medium hidden lg:table-cell">AST</th>
                <th className="text-right px-3 py-3 font-medium hidden xl:table-cell">TS%</th>
                <th className="text-right px-3 py-3 font-medium hidden xl:table-cell">BPM</th>
                <th className="px-3 py-3 w-24"></th>
              </tr>
            </thead>
            <tbody>
              {draftedPlayers.map((player, idx) => {
                const meta = ARCHETYPE_META[player.archetype] ?? {}
                const playerKey = player.name + '|' + player.team
                const logEntry = logByPlayerKey[playerKey]
                const wnbaTeam = logEntry ? teams.find((t) => t.id === logEntry.teamId) : null
                return (
                  <tr key={playerKey} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                    <td className="px-4 py-3 text-gray-500 font-mono text-sm font-bold">
                      #{logEntry ? logEntry.pick : idx + 1}
                    </td>
                    <td className="px-4 py-3">
                      <div className="font-semibold text-white">{player.name}</div>
                      <div className="text-xs text-gray-400">
                        {player.pos} · {player.team}
                      </div>
                    </td>
                    {draftLog.some(Boolean) && (
                      <td className="px-4 py-3 hidden sm:table-cell">
                        {wnbaTeam ? (
                          <div className="flex items-center gap-2">
                            <div
                              className="w-6 h-6 rounded flex items-center justify-center text-white text-xs font-bold"
                              style={{ background: wnbaTeam.primary }}
                            >
                              {wnbaTeam.abbrev.slice(0, 3)}
                            </div>
                            <span className="text-xs text-gray-400">{wnbaTeam.name}</span>
                          </div>
                        ) : (
                          <span className="text-gray-600 text-xs">—</span>
                        )}
                      </td>
                    )}
                    <td className="px-4 py-3 hidden md:table-cell">
                      <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border ${meta.bg} ${meta.border} ${meta.text}`}>
                        {meta.icon} {player.archetype}
                      </span>
                    </td>
                    <td className="px-3 py-3 text-right hidden sm:table-cell text-gray-300">{fmt(player.pts_per_g)}</td>
                    <td className="px-3 py-3 text-right hidden sm:table-cell text-gray-300">{fmt(player.treb_per_g)}</td>
                    <td className="px-3 py-3 text-right hidden lg:table-cell text-gray-300">{fmt(player.ast_per_g)}</td>
                    <td className="px-3 py-3 text-right hidden xl:table-cell text-gray-300">{pct(player.ts_pct)}</td>
                    <td className="px-3 py-3 text-right hidden xl:table-cell text-gray-300">{fmt(player.bpm)}</td>
                    <td className="px-3 py-3 text-right">
                      <button
                        onClick={() => onUndraft(player)}
                        className="px-2.5 py-1 bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white text-xs font-medium rounded transition-colors"
                      >
                        Undo
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
