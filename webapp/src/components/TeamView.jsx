import { ARCHETYPE_META } from '../data/teams'
import NeedsDisplay from './NeedsDisplay'
import PlayerTable from './PlayerTable'

export default function TeamView({ team, needs, players, playersLoaded, drafted, onDraft, onUndraft, onBack, backLabel = '← All Teams' }) {
  // Top need archetype
  const topArchetype = needs
    ? Object.entries(needs.archetypeNeeds ?? {}).sort((a, b) => b[1] - a[1])[0]?.[0]
    : null
  const topMeta = topArchetype ? ARCHETYPE_META[topArchetype] : null

  return (
    <div>
      {/* Back */}
      <button
        onClick={onBack}
        className="text-gray-500 hover:text-gray-300 text-sm mb-4 flex items-center gap-1 transition-colors"
      >
        {backLabel}
      </button>

      {/* Team header */}
      <div
        className="rounded-xl p-5 mb-6 relative overflow-hidden"
        style={{ background: `linear-gradient(135deg, ${team.primary}22 0%, ${team.secondary}11 100%)`, border: `1px solid ${team.primary}44` }}
      >
        <div className="flex items-center gap-4">
          <div
            className="w-14 h-14 rounded-xl flex items-center justify-center text-white font-bold text-xl flex-shrink-0"
            style={{ background: team.primary }}
          >
            {team.abbrev}
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">{team.name}</h1>
            {topArchetype && topMeta && (
              <p className={`text-sm mt-1 ${topMeta.text}`}>
                Top need: {topMeta.icon} {topArchetype}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
        {/* Left: Needs */}
        <div>
          <h2 className="text-base font-semibold mb-3 text-gray-200">Team Needs</h2>
          <NeedsDisplay needs={needs} teamColor={team.primary} />
        </div>

        {/* Right: Player fits */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-base font-semibold text-gray-200">Best Available Fits</h2>
            {!playersLoaded && (
              <span className="text-xs text-gray-500 animate-pulse">Loading players…</span>
            )}
            {playersLoaded && players.length === 0 && (
              <span className="text-xs text-amber-500">
                Run prepare_data.py to load player data
              </span>
            )}
          </div>
          {playersLoaded && players.length === 0 ? (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 text-center">
              <p className="text-gray-400 font-medium mb-2">No player data loaded</p>
              <p className="text-gray-600 text-sm">
                Run <code className="bg-gray-800 px-1 rounded text-gray-300">python prepare_data.py</code> from the project root to generate{' '}
                <code className="bg-gray-800 px-1 rounded text-gray-300">public/players.json</code>.
              </p>
            </div>
          ) : (
            <PlayerTable
              players={players}
              needs={needs}
              drafted={drafted}
              onDraft={onDraft}
              onUndraft={onUndraft}
              teamId={team.id}
            />
          )}
        </div>
      </div>
    </div>
  )
}
