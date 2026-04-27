import { useState, useCallback, useEffect } from 'react'
import Board from './Board'
import './PlacementPhase.css'

const SHIP_NAMES = { 5: 'Carrier', 4: 'Battleship', 3: 'Cruiser', 2: 'Destroyer' }

function shipCells(row, col, size, horiz) {
  return Array.from({ length: size }, (_, i) =>
    horiz ? { row, col: col + i } : { row: row + i, col }
  )
}

function placementValid(cells, occupied, rows, cols) {
  return cells.every(({ row, col }) =>
    row >= 0 && row < rows && col >= 0 && col < cols &&
    !occupied.has(`${row},${col}`)
  )
}

function autoPlace(sizes, rows, cols) {
  const occupied = new Set()
  const results = []
  for (const size of sizes) {
    let placed = false
    for (let t = 0; t < 400 && !placed; t++) {
      const horiz = Math.random() < 0.5
      const r = horiz
        ? Math.floor(Math.random() * rows)
        : Math.floor(Math.random() * (rows - size + 1))
      const c = horiz
        ? Math.floor(Math.random() * (cols - size + 1))
        : Math.floor(Math.random() * cols)
      const cells = shipCells(r, c, size, horiz)
      if (placementValid(cells, occupied, rows, cols)) {
        cells.forEach(({ row, col }) => occupied.add(`${row},${col}`))
        results.push({
          start_row: r, start_col: c,
          end_row: horiz ? r : r + size - 1,
          end_col: horiz ? c + size - 1 : c,
        })
        placed = true
      }
    }
    if (!placed) return null
  }
  return results
}

// Build a ships 2D matrix from confirmed placements
function buildShipsMatrix(placements, rows, cols) {
  const matrix = Array.from({ length: rows }, () => Array(cols).fill(0))
  const occupied = new Set()
  placements.forEach(({ start_row, start_col, end_row, end_col }, i) => {
    const horiz = start_row === end_row
    const size = horiz ? end_col - start_col + 1 : end_row - start_row + 1
    shipCells(start_row, start_col, size, horiz).forEach(({ row, col }) => {
      matrix[row][col] = i + 1
      occupied.add(`${row},${col}`)
    })
  })
  return { matrix, occupied }
}

export default function PlacementPhase({ shipSizes, rows = 10, cols = 10, onReady }) {
  const [placements, setPlacements] = useState([])
  const [hoverCell, setHoverCell] = useState(null)
  const [horiz, setHoriz] = useState(true)

  const currentIdx = placements.length
  const done = currentIdx >= shipSizes.length
  const currentSize = shipSizes[currentIdx]

  const { matrix: shipsMatrix, occupied } = buildShipsMatrix(placements, rows, cols)

  const ghostCells = hoverCell && !done
    ? shipCells(hoverCell.row, hoverCell.col, currentSize, horiz)
    : null
  const ghostValid = ghostCells ? placementValid(ghostCells, occupied, rows, cols) : false

  useEffect(() => {
    const handler = (e) => { if (e.key === 'r' || e.key === 'R') setHoriz(h => !h) }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const handleClick = useCallback((row, col) => {
    if (done) return
    const cells = shipCells(row, col, currentSize, horiz)
    if (!placementValid(cells, occupied, rows, cols)) return
    setPlacements(prev => [
      ...prev,
      {
        start_row: row, start_col: col,
        end_row: horiz ? row : row + currentSize - 1,
        end_col: horiz ? col + currentSize - 1 : col,
      },
    ])
  }, [done, currentSize, horiz, occupied, rows, cols])

  const EMPTY_HITS  = Array.from({ length: rows }, () => Array(cols).fill(false))
  const EMPTY_MISS  = Array.from({ length: rows }, () => Array(cols).fill(false))

  return (
    <div className="placement-phase">
      <h2>Place Your Ships</h2>
      <p className="sub">
        {done
          ? 'All ships placed — click Start Battle!'
          : `Placing: ${SHIP_NAMES[currentSize] ?? 'Ship'} (size ${currentSize}) · Press R or click to rotate`}
      </p>

      <div className="placement-layout">
        <Board
          rows={rows} cols={cols}
          hits={EMPTY_HITS} misses={EMPTY_MISS}
          ships={shipsMatrix}
          hoverCells={ghostCells}
          hoverValid={ghostValid}
          onCellClick={handleClick}
          onCellHover={(r, c) => setHoverCell({ row: r, col: c })}
          onBoardLeave={() => setHoverCell(null)}
          disabled={done}
          label="Your Board"
        />

        <div className="ship-list">
          <div className="ship-list-title">Fleet</div>
          {shipSizes.map((size, i) => (
            <div
              key={i}
              className={`ship-item ${
                i < placements.length ? 'placed' :
                i === placements.length ? 'current' : 'pending'
              }`}
            >
              <span className="ship-name">{SHIP_NAMES[size] ?? `Ship (${size})`}</span>
              <div className="ship-preview">
                {Array.from({ length: size }, (_, j) => (
                  <span key={j} className="ship-cell-icon" />
                ))}
              </div>
            </div>
          ))}

          <div className="placement-actions">
            <button className="btn-secondary" onClick={() => setHoriz(h => !h)}>
              {horiz ? '↔ Horizontal' : '↕ Vertical'}
            </button>
            <button className="btn-secondary" onClick={() => {
              const r = autoPlace(shipSizes, rows, cols)
              if (r) setPlacements(r)
            }}>
              Auto-place
            </button>
            <button
              className="btn-secondary"
              onClick={() => setPlacements(p => p.slice(0, -1))}
              disabled={placements.length === 0}
            >
              Undo
            </button>
            <button className="btn-primary" onClick={() => onReady(placements)} disabled={!done}>
              Start Battle
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
