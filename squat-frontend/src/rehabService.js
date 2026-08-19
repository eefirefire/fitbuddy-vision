// Talks to the Flask rehab backend (Scripts/rehab_knee_extension.py).
// Run it separately with: python Scripts/rehab_knee_extension.py
const REHAB_API_BASE = 'http://localhost:5050'

// All rehab routes are gated behind login (rehab_auth.py), and the login
// session lives in a cookie — fetch() does not send cookies cross-origin
// unless explicitly told to.
const WITH_SESSION = { credentials: 'include' }

// MJPEG stream — point an <img> tag at this. Browsers send cookies on plain
// <img src> requests automatically (no CORS preflight, unlike fetch), so the
// login session still applies without any extra wiring here.
export const REHAB_STREAM_URL = `${REHAB_API_BASE}/api/rehab/stream`

export async function uploadRehabVideo(videoFile) {
  const formData = new FormData()
  formData.append('video', videoFile)
  const res = await fetch(`${REHAB_API_BASE}/api/rehab/upload`, { method: 'POST', body: formData, ...WITH_SESSION })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Could not process the uploaded clip.')
  return data
}

export async function advanceRehabState() {
  const res = await fetch(`${REHAB_API_BASE}/api/rehab/advance`, { method: 'POST', ...WITH_SESSION })
  if (!res.ok) throw new Error('Could not reach rehab backend (advance).')
  return res.json()
}

export async function getRehabReport() {
  const res = await fetch(`${REHAB_API_BASE}/api/rehab/report`, WITH_SESSION)
  if (!res.ok) throw new Error('Could not reach rehab backend (report).')
  return res.json()
}

export async function resetRehabSession() {
  const res = await fetch(`${REHAB_API_BASE}/api/rehab/reset`, { method: 'POST', ...WITH_SESSION })
  if (!res.ok) throw new Error('Could not reach rehab backend (reset).')
  return res.json()
}

export async function getLiveClipSummary() {
  const res = await fetch(`${REHAB_API_BASE}/api/rehab/clip-summary`, { method: 'POST', ...WITH_SESSION })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Could not fetch live clip summary.')
  return data
}

// Polled during an active live recording to drive the rep counter, voice
// cues, and ROM gauge — see RehabSession.live_status in the backend for why
// this is a separate endpoint from the MJPEG video stream. Swallows network
// errors into `null` rather than throwing, since a single missed poll during
// a live demo shouldn't interrupt the recording — the caller just tries
// again on the next tick.
export async function getLiveStatus() {
  try {
    const res = await fetch(`${REHAB_API_BASE}/api/rehab/live-status`, WITH_SESSION)
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}
