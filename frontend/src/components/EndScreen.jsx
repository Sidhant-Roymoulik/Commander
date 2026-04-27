import Board from './Board'
import './EndScreen.css'

export default function EndScreen({ winner, playerBoard, aiBoard, onPlayAgain }) {
  const playerWon = winner === 'player'

  return (
    <div className="end-screen">
      <div className={`result-banner ${playerWon ? 'win' : 'loss'}`}>
        {playerWon ? '🎖 Victory!' : '💀 Defeated'}
      </div>
      <p className="result-sub">
        {playerWon
          ? 'You sank the enemy fleet!'
          : 'The AI found all your ships.'}
      </p>

      <div className="end-boards">
        <Board
          hits={playerBoard.hits}
          misses={playerBoard.misses}
          ships={playerBoard.ships}
          sunkCells={playerBoard.sunk}
          label="Your Board"
          disabled
        />
        <Board
          hits={aiBoard.hits}
          misses={aiBoard.misses}
          ships={aiBoard.ships}
          sunkCells={aiBoard.sunk}
          label="Enemy Board (revealed)"
          disabled
        />
      </div>

      <button className="btn-primary play-again" onClick={onPlayAgain}>
        Play Again
      </button>
    </div>
  )
}
