import { useState, useCallback } from 'react'
import { newGame, placeShips, playerAttack, aiAttack } from './api'
import PlacementPhase from './components/PlacementPhase'
import BattlePhase from './components/BattlePhase'
import EndScreen from './components/EndScreen'
import './App.css'

const ROWS = 10, COLS = 10

function emptyBoard() {
  return {
    hits:   Array.from({ length: ROWS }, () => Array(COLS).fill(false)),
    misses: Array.from({ length: ROWS }, () => Array(COLS).fill(false)),
    ships:  null,
  }
}

export default function App() {
  const [phase, setPhase]         = useState('home')       // home | placement | playing | finished
  const [sessionId, setSessionId] = useState(null)
  const [shipSizes, setShipSizes] = useState([5, 4, 3, 3, 2])
  const [playerBoard, setPlayer]  = useState(emptyBoard())
  const [aiBoard, setAi]          = useState(emptyBoard())
  const [aiThinking, setAiThink]  = useState(false)
  const [lastAiAttack, setLastAi] = useState(null)
  const [probMap, setProbMap]     = useState(null)
  const [winner, setWinner]       = useState(null)
  const [message, setMessage]     = useState('Your turn — click a cell to fire.')

  const startGame = async () => {
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
  }

  const handlePlacementReady = useCallback(async (placements) => {
    const data = await placeShips(sessionId, placements)
    setPlayer({
      hits:   data.player_board.hits,
      misses: data.player_board.misses,
      ships:  data.player_board.ships,
    })
    setPhase('playing')
  }, [sessionId])

  const handlePlayerAttack = useCallback(async (row, col) => {
    if (aiThinking) return

    // Player fires
    let data
    try {
      data = await playerAttack(sessionId, row, col)
    } catch {
      return // already attacked cell — silently ignore
    }

    setAi({
      hits:   data.ai_board.hits,
      misses: data.ai_board.misses,
      ships:  data.ai_board.ships,
    })
    setMessage(data.hit ? '💥 Hit!' : '〇 Miss.')

    if (data.game_over) {
      setWinner(data.winner)
      setPhase('finished')
      return
    }

    // AI fires
    setAiThink(true)
    await new Promise(r => setTimeout(r, 600)) // brief pause for drama
    const aiData = await aiAttack(sessionId)
    setAiThink(false)

    setLastAi({ row: aiData.row, col: aiData.col })
    setProbMap(aiData.prob_map)
    setPlayer({
      hits:   aiData.player_board.hits,
      misses: aiData.player_board.misses,
      ships:  aiData.player_board.ships,
    })

    if (aiData.game_over) {
      setWinner(aiData.winner)
      setPhase('finished')
    } else {
      setMessage(aiData.hit
        ? `AI hit your ship at ${String.fromCharCode(65 + aiData.col)}${aiData.row + 1}! Your turn.`
        : `AI missed at ${String.fromCharCode(65 + aiData.col)}${aiData.row + 1}. Your turn.`
      )
    }
  }, [sessionId, aiThinking])

  return (
    <div className="app">
      <header className="app-header">
        <span className="logo-text">⚓ Commander</span>
        <span className="logo-sub">Battleship vs AI</span>
      </header>

      {phase === 'home' && (
        <div className="home">
          <h1>Commander</h1>
          <p className="home-desc">
            Play Battleship against a CNN trained on 100k games.
            The AI uses probability maps to hunt your fleet.
          </p>
          <button className="btn-primary home-btn" onClick={startGame}>
            New Game
          </button>
        </div>
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
          onPlayAgain={startGame}
        />
      )}
    </div>
  )
}
