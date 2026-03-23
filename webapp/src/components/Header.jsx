export default function Header({ view, selectedTeam, draftedCount, onHome, onBoard }) {
  return (
    <header className="border-b border-gray-800 bg-[#0D1120] sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
        <button
          onClick={onHome}
          className="flex items-center gap-2 hover:opacity-80 transition-opacity"
        >
          <span className="text-[#FF6B2B] font-bold text-xl">WNBA</span>
          <span className="text-white font-semibold text-lg hidden sm:block">Draft Model</span>
          <span className="text-gray-500 text-sm hidden sm:block">· 2025</span>
        </button>

        <nav className="flex items-center gap-2">
          <button
            onClick={onHome}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              view === 'teams'
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            Teams
          </button>
          <button
            onClick={onBoard}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors flex items-center gap-1.5 ${
              view === 'board'
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            Draft Board
            {draftedCount > 0 && (
              <span className="bg-[#FF6B2B] text-white text-xs font-bold px-1.5 py-0.5 rounded-full">
                {draftedCount}
              </span>
            )}
          </button>
        </nav>
      </div>
    </header>
  )
}
