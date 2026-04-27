import { useState, useCallback, useEffect } from 'react'
import { newGame, placeShips, playerAttack, aiAttack } from './api'
import PlacementPhase from './components/PlacementPhase'
import BattlePhase from './components/BattlePhase'
import EndScreen from './components/EndScreen'
import './App.css'

const ROWS = 10, COLS = 10

const SHIP_NAMES = { 5: 'Carrier', 4: 'Battleship', 3: 'Cruiser', 2: 'Destroyer' }

function emptyBoard() {
  return {
    hits:   Array.from({ length: ROWS }, () => Array(COLS).fill(false)),
    misses: Array.from({ length: ROWS }, () => Array(COLS).fill(false)),
    ships:  null,
    sunk:   new Set(),
  }
}

// Convert [[r,c],...] list from API into a Set of "r,c" strings for O(1) lookup
function sunkSet(cells) {
  return new Set((cells ?? []).map(([r, c]) => `${r},${c}`))
}

function boardFromData(data) {
  return {
    hits:   data.hits,
    misses: data.misses,
    ships:  data.ships,
    sunk:   sunkSet(data.sunk),
  }
}

// Infer ship size from just_sunk cell list
function sunkShipName(cells) {
  return SHIP_NAMES[cells.length] ?? `size-${cells.length} ship`
}

export default function App() {
  const [phase, setPhase]         = useState('loading')
  const [sessionId, setSessionId] = useState(null)
  const [shipSizes, setShipSizes] = useState([5, 4, 3, 3, 2])
  const [playerBoard, setPlayer]  = useState(emptyBoard())
  const [aiBoard, setAi]          = useState(emptyBoard())
  const [aiThinking, setAiThink]  = useState(false)
  const [lastAiAttack, setLastAi] = useState(null)
  const [probMap, setProbMap]     = useState(null)
  const [winner, setWinner]       = useState(null)
  const [message, setMessage]     = useState('Your turn — click a cell to fire.')

  const startGame = useCallback(async () => {
    setPhase('loading')
    const data = await newGame()
    setSessionId(data.session_id)
    setShipSizes(data.ship_sizes)
    setPlayer(emptyBoard())
    setAi(emptyBoard())
    setLastAi(null)
    setProbMap(null)
    setWinner(null)
    setMessage('Your turn — click a cell to fire.')
    setPhase('placement')
  }, [])

  useEffect(() => { startGame() }, [startGame])

  const handlePlacementReady = useCallback(async (placements) => {
    const data = await placeShips(sessionId, placements)
    setPlayer(boardFromData(data.player_board))
    setPhase('playing')
  }, [sessionId])

  const handlePlayerAttack = useCallback(async (row, col) => {
    if (aiThinking) return

    let data
    try {
      data = await playerAttack(sessionId, row, col)
    } catch {
      return
    }

    setAi(boardFromData(data.ai_board))

    let msg = data.hit ? '💥 Hit!' : '〇 Miss.'
    if (data.just_sunk?.length) {
      msg = `🔥 You sunk the enemy ${sunkShipName(data.just_sunk)}!`
    }
    setMessage(msg)

    if (data.game_over) {
      setWinner(data.winner)
      setPhase('finished')
      return
    }

    setAiThink(true)
    await new Promise(r => setTimeout(r, 700))
    const aiData = await aiAttack(sessionId)
    setAiThink(false)

    setLastAi({ row: aiData.row, col: aiData.col })
    setProbMap(aiData.prob_map)
    setPlayer(boardFromData(aiData.player_board))

    if (aiData.game_over) {
      setWinner(aiData.winner)
      setPhase('finished')
    } else {
      const coord = `${String.fromCharCode(65 + aiData.col)}${aiData.row + 1}`
      if (aiData.just_sunk?.length) {
        setMessage(`💀 AI sunk your ${sunkShipName(aiData.just_sunk)}! Your turn.`)
      } else if (aiData.hit) {
        setMessage(`AI hit your ship at ${coord}! Your turn.`)
      } else {
        setMessage(`AI missed at ${coord}. Your turn.`)
      }
    }
  }, [sessionId, aiThinking])

  return (
    <div className="app">
      <header className="app-header">
        <span className="logo-text">⚓ Commander</span>
        <span className="logo-sub">Battleship vs AI</span>
        {phase === 'playing' && (
          <button className="btn-secondary new-game-btn" onClick={startGame}>New Game</button>
        )}
      </header>

      {phase === 'loading' && (
        <div className="loading">Starting game…</div>
      )}

      {phase === 'placement' && (
        <PlacementPhase
          shipSizes={shipSizes}
          rows={ROWS}
          cols={COLS}
          onReady={handlePlacementReady}
        />
      )}

      {phase === 'playing' && (
        <BattlePhase
          playerBoard={playerBoard}
          aiBoard={aiBoard}
          onPlayerAttack={handlePlayerAttack}
          aiThinking={aiThinking}
          lastAiAttack={lastAiAttack}
          probMap={probMap}
          message={message}
        />
      )}

      {phase === 'finished' && (
        <EndScreen
          winner={winner}
          playerBoard={playerBoard}
          aiBoard={aiBoard}
          onPlayAgain={() => startGame()}
        />
      )}
    </div>
  )
}
