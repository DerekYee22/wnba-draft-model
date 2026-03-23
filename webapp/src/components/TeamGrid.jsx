export default function TeamGrid({ teams, onSelect }) {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-1">Select a Team</h1>
      <p className="text-gray-400 text-sm mb-6">
        View team needs and ranked player fits for the 2025 WNBA Draft.
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {teams.map((team) => (
          <button
            key={team.id}
            onClick={() => onSelect(team)}
            className="group relative rounded-xl border border-gray-800 bg-gray-900 p-4 text-left hover:border-gray-600 transition-all hover:scale-[1.02] hover:shadow-lg"
            style={{ '--team-color': team.primary }}
          >
            {/* Color bar */}
            <div
              className="absolute top-0 left-0 right-0 h-1 rounded-t-xl"
              style={{ background: team.primary }}
            />
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold text-sm mb-3 mt-1"
              style={{ background: team.primary }}
            >
              {team.abbrev}
            </div>
            <div className="text-white font-semibold text-sm leading-tight">
              {team.name}
            </div>
            <div className="mt-2 text-xs text-gray-500 group-hover:text-gray-400 transition-colors">
              View needs →
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
