import { useState, useEffect } from 'react'
import { TEAMS } from './data/teams'
import { DEFAULT_TEAM_NEEDS } from './data/defaultNeeds'
import Header from './components/Header'
import TeamGrid from './components/TeamGrid'
import TeamView from './components/TeamView'
import DraftBoard from './components/DraftBoard'
import MockDraft from './components/MockDraft'

const DRAFTED_KEY = 'wnba_draft_2026_drafted'

// Actual 2026 WNBA Draft order (45 picks across 3 rounds, including traded picks)
const DRAFT_ORDER_2026 = [
  // Round 1
  'dallas-wings',            // 1
  'minnesota-lynx',          // 2  (from Chicago)
  'seattle-storm',           // 3  (from LA Sparks)
  'washington-mystics',      // 4
  'chicago-sky',             // 5  (from Connecticut)
  'toronto-tempo',           // 6
  'portland-fire',           // 7
  'golden-state-valkyries',  // 8
  'washington-mystics',      // 9  (from Seattle)
  'indiana-fever',           // 10
  'washington-mystics',      // 11 (from NY via MIN/CON)
  'connecticut-sun',         // 12 (from Phoenix via Chicago)
  'atlanta-dream',           // 13
  'seattle-storm',           // 14 (from Las Vegas)
  'connecticut-sun',         // 15 (from Minnesota via Washington)
  // Round 2
  'seattle-storm',           // 16 (from Dallas)
  'portland-fire',           // 17 (from Chicago)
  'connecticut-sun',         // 18
  'washington-mystics',      // 19
  'los-angeles-sparks',      // 20
  'chicago-sky',             // 21 (from Portland)
  'toronto-tempo',           // 22
  'golden-state-valkyries',  // 23
  'los-angeles-sparks',      // 24 (from Seattle)
  'indiana-fever',           // 25
  'toronto-tempo',           // 26 (from NY via Chicago)
  'phoenix-mercury',         // 27
  'atlanta-dream',           // 28
  'las-vegas-aces',          // 29
  'washington-mystics',      // 30 (from Minnesota)
  // Round 3
  'dallas-wings',            // 31
  'chicago-sky',             // 32
  'connecticut-sun',         // 33
  'washington-mystics',      // 34
  'los-angeles-sparks',      // 35
  'toronto-tempo',           // 36
  'portland-fire',           // 37
  'golden-state-valkyries',  // 38
  'seattle-storm',           // 39
  'indiana-fever',           // 40
  'new-york-liberty',        // 41
  'phoenix-mercury',         // 42
  'atlanta-dream',           // 43
  'las-vegas-aces',          // 44
  'minnesota-lynx',          // 45
]

export default function App() {
  const [view, setView] = useState('teams') // 'teams' | 'team' | 'board' | 'mock'
  const [selectedTeam, setSelectedTeam] = useState(null)
  const [teamViewSource, setTeamViewSource] = useState('teams') // 'teams' | 'mock'
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

  // Mock draft state
  const [mockDraftOrder, setMockDraftOrder] = useState([])   // array of team IDs (39 picks)
  const [mockCurrentPick, setMockCurrentPick] = useState(0)  // 0-indexed
  const [mockDraftLog, setMockDraftLog] = useState([])        // [{pick, teamId, player}] indexed by pick

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
        setPlayersLoaded(true)
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
      .catch(() => {})
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

  // Mock draft functions
  function startMockDraft() {
    setMockDraftOrder(DRAFT_ORDER_2026)
    setMockCurrentPick(0)
    setMockDraftLog([])
    setDrafted(new Set())
    setView('mock')
  }

  function mockDraftPlayer(player) {
    const key = player.name + '|' + player.team
    setDrafted((prev) => new Set([...prev, key]))
    const teamId = mockDraftOrder[mockCurrentPick]
    setMockDraftLog((prev) => {
      const next = [...prev]
      next[mockCurrentPick] = { pick: mockCurrentPick + 1, teamId, player }
      return next
    })
    setMockCurrentPick((prev) => prev + 1)
  }

  function resetMockDraft() {
    startMockDraft()
  }

  function selectTeam(team, source = 'teams') {
    setSelectedTeam(team)
    setTeamViewSource(source)
    setView('team')
  }

  function goHome() {
    setView('teams')
    setSelectedTeam(null)
  }

  function goBackFromTeam() {
    if (teamViewSource === 'mock') {
      setView('mock')
      setSelectedTeam(null)
    } else {
      goHome()
    }
  }

  return (
    <div className="min-h-screen bg-[#0A0E1A] text-gray-100">
      <Header
        view={view}
        selectedTeam={selectedTeam}
        draftedCount={drafted.size}
        hasMockDraft={mockDraftOrder.length > 0}
        mockCurrentPick={mockCurrentPick}
        mockTotal={mockDraftOrder.length}
        onHome={goHome}
        onBoard={() => setView('board')}
        onMock={() => (mockDraftOrder.length > 0 ? setView('mock') : startMockDraft())}
        onStartMock={startMockDraft}
      />
      <main className="max-w-7xl mx-auto px-4 py-6">
        {view === 'teams' && (
          <TeamGrid teams={TEAMS} onSelect={(t) => selectTeam(t, 'teams')} />
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
            onBack={goBackFromTeam}
            backLabel={teamViewSource === 'mock' ? '← Back to Draft' : '← All Teams'}
          />
        )}
        {view === 'board' && (
          <DraftBoard
            players={players}
            drafted={drafted}
            draftLog={mockDraftLog}
            teams={TEAMS}
            onUndraft={undraftPlayer}
            onReset={resetDraft}
          />
        )}
        {view === 'mock' && (
          <MockDraft
            draftOrder={mockDraftOrder}
            currentPick={mockCurrentPick}
            draftLog={mockDraftLog}
            teams={TEAMS}
            teamNeeds={teamNeeds}
            players={players}
            playersLoaded={playersLoaded}
            drafted={drafted}
            onDraft={mockDraftPlayer}
            onViewTeam={(team) => selectTeam(team, 'mock')}
          />
        )}
      </main>
    </div>
  )
}
