const REHAB_API_BASE = 'http://localhost:5050'
const WITH_SESSION = { credentials: 'include' }

async function postJson(path, body) {
  const res = await fetch(`${REHAB_API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    ...WITH_SESSION,
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Request failed.')
  return data
}

export function registerAccount(username, password) {
  return postJson('/api/auth/register', { username, password })
}

export function login(username, password) {
  return postJson('/api/auth/login', { username, password })
}

export async function logout() {
  const res = await fetch(`${REHAB_API_BASE}/api/auth/logout`, { method: 'POST', ...WITH_SESSION })
  return res.json()
}

export async function getCurrentUser() {
  const res = await fetch(`${REHAB_API_BASE}/api/auth/me`, WITH_SESSION)
  if (!res.ok) return null
  return res.json()
}

export function submitIntake(intake) {
  return postJson('/api/intake', intake)
}

export async function getCurrentIntake() {
  const res = await fetch(`${REHAB_API_BASE}/api/intake/current`, WITH_SESSION)
  if (!res.ok) return null
  return res.json()
}
