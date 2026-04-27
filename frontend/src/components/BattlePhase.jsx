import Board from './Board'
import './BattlePhase.css'

export default function BattlePhase({
  playerBoard,
  aiBoard,
  onPlayerAttack,
  aiThinking,
  lastAiAttack,
  probMap,
  message,
}) {
  return (
    <div className="battle-phase">
      <div className="status-bar">
        <span className={`status-msg ${aiThinking ? 'thinking' : ''}`}>
          {aiThinking ? '⚡ AI is thinking…' : message}
        </span>
      </div>

      <div className="boards-row">
        <div className="board-section">
          <Board
            hits={playerBoard.hits}
            misses={playerBoard.misses}
            ships={playerBoard.ships}
            highlightCell={lastAiAttack}
            probMap={probMap}
            label="Your Board"
            disabled
          />
          <p className="board-hint">AI's attacks — your ships are shown</p>
        </div>

        <div className="divider">VS</div>

        <div className="board-section">
          <Board
            hits={aiBoard.hits}
            misses={aiBoard.misses}
            ships={aiBoard.ships}   // null until game over
            onCellClick={onPlayerAttack}
            disabled={aiThinking}
            label="Enemy Waters"
          />
          <p className="board-hint">
            {aiThinking ? 'Waiting for AI…' : 'Click to fire'}
          </p>
        </div>
      </div>
    </div>
  )
}
