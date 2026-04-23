import { useEffect, useRef } from 'react'
import { ARCHETYPE_META } from '../data/teams'
import PlayerTable from './PlayerTable'

function CompactNeeds({ needs, teamColor }) {
  if (!needs?.archetypeNeeds) return null
  const sorted = Object.entries(needs.archetypeNeeds).sort((a, b) => b[1] - a[1])
  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {sorted.map(([arch, score], idx) => {
        const meta = ARCHETYPE_META[arch]
        if (!meta) return null
        return (
          <span
            key={arch}
            className={`inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full border ${meta.bg} ${meta.border} ${meta.text}`}
          >
            {meta.icon} {arch}
            <span className="opacity-60 ml-0.5">{score}/10</span>
            {idx === 0 && (
              <span className="ml-1 text-[10px] bg-orange-500/30 text-orange-300 px-1 rounded">top</span>
            )}
          </span>
        )
      })}
    </div>
  )
}

export default function MockDraft({
  draftOrder,
  currentPick,
  draftLog,
  teams,
  teamNeeds,
  players,
  playersLoaded,
  drafted,
  onDraft,
  onViewTeam,
}) {
  const numTeams = teams.length
  const isDone = currentPick >= draftOrder.length
  const currentTeamId = !isDone ? draftOrder[currentPick] : null
  const currentTeam = teams.find((t) => t.id === currentTeamId)
  const currentNeeds = currentTeam ? teamNeeds[currentTeam.name] : null
  const round = Math.floor(currentPick / numTeams) + 1
  const pickInRound = (currentPick % numTeams) + 1

  const currentPickRef = useRef(null)
  useEffect(() => {
    currentPickRef.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [currentPick])

  return (
    <div className="flex gap-4 items-start min-w-0">
      {/* Sidebar: Draft Order */}
      <div className="w-48 flex-shrink-0 sticky top-20">
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
            <span className="text-xs text-gray-500 uppercase tracking-wide font-medium">Order</span>
            <span className="text-xs text-gray-600">
              {Math.min(currentPick, draftOrder.length)}/{draftOrder.length}
            </span>
          </div>
          <div className="overflow-y-auto" style={{ maxHeight: 'calc(100vh - 190px)' }}>
            {draftOrder.map((teamId, idx) => {
              const team = teams.find((t) => t.id === teamId)
              const logEntry = draftLog[idx]
              const isCurrent = idx === currentPick
              const isPast = idx < currentPick
              const roundNum = Math.floor(idx / numTeams) + 1
              const showRoundHeader = idx % numTeams === 0

              return (
                <div key={idx}>
                  {showRoundHeader && (
                    <div className="px-3 py-1 bg-gray-800/50 text-xs text-gray-500 uppercase tracking-wide font-medium">
                      Round {roundNum}
                    </div>
                  )}
                  <div
                    ref={isCurrent ? currentPickRef : null}
                    className={`px-2 py-1.5 flex items-center gap-2 border-b border-gray-800/30 transition-colors
                      ${isCurrent ? 'bg-gray-800/80' : ''}
                      ${isPast ? 'opacity-35' : ''}
                    `}
                  >
                    <span className="text-gray-600 text-xs w-5 text-right flex-shrink-0 font-mono">
                      {idx + 1}
                    </span>
                    <button
                      onClick={() => onViewTeam(team)}
                      className="w-6 h-6 rounded flex items-center justify-center text-white text-xs font-bold flex-shrink-0 hover:opacity-75 transition-opacity"
                      style={{ background: team?.primary }}
                      title={`View ${team?.name}`}
                    >
                      {team?.abbrev?.slice(0, 3)}
                    </button>
                    <div className="flex-1 min-w-0">
                      {logEntry ? (
                        <span className="text-gray-400 text-xs truncate block leading-tight">
                          {logEntry.player.name}
                        </span>
                      ) : isCurrent ? (
                        <span className="text-white text-xs font-semibold">Clock</span>
                      ) : (
                        <span className="text-gray-600 text-xs">{team?.abbrev}</span>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Main area */}
      <div className="flex-1 min-w-0 overflow-hidden">
        {isDone ? (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-16 text-center">
            <p className="text-3xl font-bold text-white mb-3">Draft Complete</p>
            <p className="text-gray-400 text-sm">
              All {draftOrder.length} picks have been made across {Math.floor(draftOrder.length / numTeams)} rounds.
            </p>
          </div>
        ) : (
          <>
            {/* On the clock header with compact needs */}
            <div
              className="rounded-xl p-5 mb-4"
              style={{
                background: `linear-gradient(135deg, ${currentTeam?.primary}22 0%, ${currentTeam?.secondary}11 100%)`,
                border: `1px solid ${currentTeam?.primary}44`,
              }}
            >
              <div className="flex items-center gap-4">
                <div
                  className="w-14 h-14 rounded-xl flex items-center justify-center text-white font-bold text-xl flex-shrink-0"
                  style={{ background: currentTeam?.primary }}
                >
                  {currentTeam?.abbrev}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-xs text-gray-500 mb-0.5 uppercase tracking-wide">
                    Pick {currentPick + 1} · Round {round} · Pick {pickInRound} of {numTeams}
                  </div>
                  <h1 className="text-xl font-bold text-white">{currentTeam?.name}</h1>
                  <CompactNeeds needs={currentNeeds} teamColor={currentTeam?.primary} />
                </div>
              </div>
            </div>

            {/* Player table full width */}
            <PlayerTable
              players={players}
              needs={currentNeeds}
              drafted={drafted}
              onDraft={onDraft}
              teamId={currentTeamId}
            />
          </>
        )}
      </div>
    </div>
  )
}
