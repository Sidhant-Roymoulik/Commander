const BASE = '/game'

async function json(res) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? res.statusText)
  }
  return res.json()
}

export const newGame = () =>
  fetch(`${BASE}/new`, { method: 'POST' }).then(json)

export const placeShips = (sessionId, ships) =>
  fetch(`${BASE}/${sessionId}/place-ships`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ships }),
  }).then(json)

export const playerAttack = (sessionId, row, col) =>
  fetch(`${BASE}/${sessionId}/player-attack`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ row, col }),
  }).then(json)

export const aiAttack = (sessionId) =>
  fetch(`${BASE}/${sessionId}/ai-attack`).then(json)
