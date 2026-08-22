// Talks to the standalone sit-to-stand Flask backend (Scripts/rehab_sit_to_stand.py),
// deliberately a separate process/port from the knee-extension backend (5050) —
// see that module's own docstring for why it's a fully self-contained file.
// Run it separately with: python Scripts/rehab_sit_to_stand.py
const SIT_TO_STAND_API_BASE = 'http://localhost:5052'

// Login/intake still lives on the knee-extension backend (rehab_auth.py,
// authService.js) — both Flask apps import the SAME rehab_auth.py module,
// share the same SQLite user database (Scripts/fitbuddy_users.db) and, as
// long as REHAB_SECRET_KEY isn't set to different values for each process,
// the same signed session cookie. Cookies are scoped by domain ("localhost"),
// not port, so logging in once against 5050 is enough for 5052 to recognize
// the same session too.
const WITH_SESSION = { credentials: 'include' }

// MJPEG stream — point an <img> tag at this, same pattern as REHAB_STREAM_URL.
export const SIT_TO_STAND_STREAM_URL = `${SIT_TO_STAND_API_BASE}/api/sit-to-stand/stream`

export async function uploadSitToStandVideo(videoFile) {
  const formData = new FormData()
  formData.append('video', videoFile)
  const res = await fetch(`${SIT_TO_STAND_API_BASE}/api/sit-to-stand/upload`, { method: 'POST', body: formData, ...WITH_SESSION })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Could not process the uploaded clip.')
  return data
}

export async function advanceSitToStandState() {
  const res = await fetch(`${SIT_TO_STAND_API_BASE}/api/sit-to-stand/advance`, { method: 'POST', ...WITH_SESSION })
  if (!res.ok) throw new Error('Could not reach sit-to-stand backend (advance).')
  return res.json()
}

export async function getSitToStandReport() {
  const res = await fetch(`${SIT_TO_STAND_API_BASE}/api/sit-to-stand/report`, WITH_SESSION)
  if (!res.ok) throw new Error('Could not reach sit-to-stand backend (report).')
  return res.json()
}

export async function resetSitToStandSession() {
  const res = await fetch(`${SIT_TO_STAND_API_BASE}/api/sit-to-stand/reset`, { method: 'POST', ...WITH_SESSION })
  if (!res.ok) throw new Error('Could not reach sit-to-stand backend (reset).')
  return res.json()
}

export async function getSitToStandClipSummary() {
  const res = await fetch(`${SIT_TO_STAND_API_BASE}/api/sit-to-stand/clip-summary`, { method: 'POST', ...WITH_SESSION })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Could not fetch live clip summary.')
  return data
}

// Polled during an active live recording — same shape/purpose as
// rehabService.getLiveStatus, see that function's comment for why
// sinceCueSeq/sinceAlignmentSeq matter (draining cue_pending/
// alignment_cue_pending, not just the single "current" cue value).
export async function getSitToStandLiveStatus(sinceCueSeq = 0, sinceAlignmentSeq = 0) {
  try {
    const url = `${SIT_TO_STAND_API_BASE}/api/sit-to-stand/live-status?since_cue_seq=${sinceCueSeq}&since_alignment_seq=${sinceAlignmentSeq}`
    const res = await fetch(url, WITH_SESSION)
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}
