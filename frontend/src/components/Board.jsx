import { Fragment } from 'react'
import './Board.css'

const COL_LABELS = 'ABCDEFGHIJ'.split('')

export default function Board({
  rows = 10,
  cols = 10,
  hits,
  misses,
  ships,            // 2D int matrix (>0 = ship) or null (hidden)
  onCellClick,
  onCellHover,      // (row, col) | null — called on mouseenter
  onBoardLeave,     // () => void — called when mouse leaves the whole grid
  disabled = false,
  hoverCells = null,    // [{row,col}] ghost placement preview
  hoverValid = true,
  highlightCell = null, // {row,col} most recent AI attack
  probMap = null,       // 2D float [0,1] heatmap overlay
  label = '',
}) {
  const cells = []

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const isHit   = !!hits?.[r]?.[c]
      const isMiss  = !!misses?.[r]?.[c]
      const hasShip = (ships?.[r]?.[c] ?? 0) > 0
      const isHL    = highlightCell?.row === r && highlightCell?.col === c
      const isHover = hoverCells?.some(h => h.row === r && h.col === c)
      const prob    = probMap?.[r]?.[c] ?? 0

      const cls = ['cell']
      if (isHit)        cls.push('hit')
      else if (isMiss)  cls.push('miss')
      else if (hasShip) cls.push('ship')
      else              cls.push('water')
      if (isHL)    cls.push('highlighted')
      if (isHover) cls.push(hoverValid ? 'hover-valid' : 'hover-invalid')
      if (!disabled && onCellClick && !isHit && !isMiss) cls.push('clickable')

      cells.push(
        <div
          key={`${r}-${c}`}
          className={cls.join(' ')}
          style={probMap && !isHit && !isMiss ? { '--prob': prob } : {}}
          onClick={() => !disabled && onCellClick?.(r, c)}
          onMouseEnter={() => onCellHover?.(r, c)}
        >
          {isHit  && <span className="marker x-mark">✕</span>}
          {isMiss && <span className="marker dot">•</span>}
        </div>
      )
    }
  }

  return (
    <div className="board-wrap">
      {label && <div className="board-label">{label}</div>}
      <div
        className="board-grid"
        style={{ '--cols': cols, '--rows': rows }}
        onMouseLeave={onBoardLeave}
      >
        <div className="corner" />
        {COL_LABELS.slice(0, cols).map(l => (
          <div key={l} className="axis-label">{l}</div>
        ))}
        {Array.from({ length: rows }, (_, r) => (
          <Fragment key={r}>
            <div className="axis-label">{r + 1}</div>
            {cells.slice(r * cols, r * cols + cols)}
          </Fragment>
        ))}
      </div>
    </div>
  )
}
