import { useState, useEffect } from 'react'
import { TEAMS } from './data/teams'
import { DEFAULT_TEAM_NEEDS } from './data/defaultNeeds'
import Header from './components/Header'
import TeamGrid from './components/TeamGrid'
import TeamView from './components/TeamView'
import DraftBoard from './components/DraftBoard'

const DRAFTED_KEY = 'wnba_draft_2025_drafted'

export default function App() {
  const [view, setView] = useState('teams') // 'teams' | 'team' | 'board'
  const [selectedTeam, setSelectedTeam] = useState(null)
  const [players, setPlayers] = useState([])
  const [teamNeeds, setTeamNeeds] = useState(DEFAULT_TEAM_NEEDS)
  const [drafted, setDrafted] = useState(() => {
    try {
      return new Set(JSON.parse(localStorage.getItem(DRAFTED_KEY) || '[]'))
    } catch {
      return new Set()
    }
  })
  const [playersLoaded, setPlayersLoaded] = useState(false)

  // Load players.json
  useEffect(() => {
    fetch('/players.json')
      .then((r) => {
        if (!r.ok) throw new Error('not found')
        return r.json()
      })
      .then((data) => {
        setPlayers(data)
        setPlayersLoaded(true)
      })
      .catch(() => {
        setPlayersLoaded(true) // no data, show empty state
      })
  }, [])

  // Load team_needs.json if available (overrides defaults)
  useEffect(() => {
    fetch('/team_needs.json')
      .then((r) => {
        if (!r.ok) throw new Error('not found')
        return r.json()
      })
      .then((data) => setTeamNeeds(data))
      .catch(() => {}) // fall back to defaults silently
  }, [])

  // Persist drafted to localStorage
  useEffect(() => {
    localStorage.setItem(DRAFTED_KEY, JSON.stringify([...drafted]))
  }, [drafted])

  function draftPlayer(player) {
    const key = player.name + '|' + player.team
    setDrafted((prev) => new Set([...prev, key]))
  }

  function undraftPlayer(player) {
    const key = player.name + '|' + player.team
    setDrafted((prev) => {
      const next = new Set(prev)
      next.delete(key)
      return next
    })
  }

  function resetDraft() {
    setDrafted(new Set())
  }

  function selectTeam(team) {
    setSelectedTeam(team)
    setView('team')
  }

  function goHome() {
    setView('teams')
    setSelectedTeam(null)
  }

  return (
    <div className="min-h-screen bg-[#0A0E1A] text-gray-100">
      <Header
        view={view}
        selectedTeam={selectedTeam}
        draftedCount={drafted.size}
        onHome={goHome}
        onBoard={() => setView('board')}
      />
      <main className="max-w-7xl mx-auto px-4 py-6">
        {view === 'teams' && (
          <TeamGrid teams={TEAMS} onSelect={selectTeam} />
        )}
        {view === 'team' && selectedTeam && (
          <TeamView
            team={selectedTeam}
            needs={teamNeeds[selectedTeam.name]}
            players={players}
            playersLoaded={playersLoaded}
            drafted={drafted}
            onDraft={draftPlayer}
            onUndraft={undraftPlayer}
            onBack={goHome}
          />
        )}
        {view === 'board' && (
          <DraftBoard
            players={players}
            drafted={drafted}
            onUndraft={undraftPlayer}
            onReset={resetDraft}
          />
        )}
      </main>
    </div>
  )
}
