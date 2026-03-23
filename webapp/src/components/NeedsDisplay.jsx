import { ARCHETYPE_META, ARCHETYPES } from '../data/teams'

export default function NeedsDisplay({ needs, teamColor }) {
  if (!needs) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <p className="text-gray-500 text-sm">No needs data available. Run the model to generate team_needs.json.</p>
      </div>
    )
  }

  // Sort archetypes by need score descending
  const sorted = ARCHETYPES.slice().sort(
    (a, b) => (needs.archetypeNeeds?.[b] ?? 0) - (needs.archetypeNeeds?.[a] ?? 0)
  )

  return (
    <div className="space-y-4">
      {/* Notes */}
      {needs.notes && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-sm text-gray-300">
          <span className="text-gray-500 font-medium">Scouting note: </span>
          {needs.notes}
        </div>
      )}

      {/* Archetype needs bars */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">
          Archetype Needs
        </h3>
        {sorted.map((arch, idx) => {
          const score = needs.archetypeNeeds?.[arch] ?? 0
          const meta = ARCHETYPE_META[arch]
          const pct = (score / 10) * 100

          return (
            <div key={arch}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-base">{meta.icon}</span>
                  <span className={`text-sm font-medium ${meta.text}`}>{arch}</span>
                  {idx === 0 && (
                    <span className="text-xs bg-orange-500/20 text-orange-400 border border-orange-500/30 px-1.5 py-0.5 rounded">
                      Top Need
                    </span>
                  )}
                </div>
                <span className="text-sm font-bold text-white">{score}/10</span>
              </div>
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{ width: `${pct}%`, background: meta.color }}
                />
              </div>
              <p className="text-xs text-gray-600 mt-1">{meta.description}</p>
            </div>
          )
        })}
      </div>

      {/* Top stat needs */}
      {needs.topStatNeeds?.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
            Statistical Gaps
          </h3>
          <div className="flex flex-wrap gap-2">
            {needs.topStatNeeds.map((stat) => (
              <span
                key={stat}
                className="px-3 py-1 rounded-full bg-gray-800 border border-gray-700 text-sm text-gray-300"
              >
                {stat}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
