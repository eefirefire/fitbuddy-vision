import { useState, useRef, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  uploadSitToStandVideo, advanceSitToStandState, getSitToStandReport, resetSitToStandSession,
  getSitToStandClipSummary, getSitToStandLiveStatus, SIT_TO_STAND_STREAM_URL,
} from '../sitToStandService'
import { getCurrentUser, getCurrentIntake } from '../authService'
import {
  classifyCameraAlignment, classifyEccentricPacing, classifyLiveRom,
  classifyStandingDeficit, classifyJerkyMotion, classifyTrackingQuality,
} from '../sitToStandInterpret'

// Same polling cadence as RehabPanel — see that component's comment for why
// (no websocket in this stack; 400ms keeps the rep counter/gauge/voice cues
// feeling live without hammering the backend's Flask dev server, which is
// also serving the MJPEG stream at the same time).
const LIVE_STATUS_POLL_MS = 400

const INTAKE_WARNING_COLOR = { red: '#e06c75', amber: '#e5a14a' }

// Sit-to-stand is bilateral (both legs rise together) — no per-leg select/
// switch flow, so this state machine is just SELECT -> RECORD -> ANALYZE,
// unlike the knee-extension module's left/right split.
const STATE_COPY = {
  STATE_SELECT: { title: 'Get Ready', hint: 'Sit side-on to the camera, in a chair, both legs visible. Click "Next" when ready.' },
  STATE_RECORD: { title: 'Recording', hint: 'Perform your sit-to-stand reps now.' },
  STATE_ANALYZE: { title: 'Analysis Ready', hint: 'Click "Get Report" to see your session summary.' },
}

const RECORD_STATES = ['STATE_RECORD']

export default function SitToStandPanel() {
  const navigate = useNavigate()
  const location = useLocation()
  const [state, setState] = useState('STATE_SELECT')
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)
  const [videoSrc, setVideoSrc] = useState(null)
  const [videoFile, setVideoFile] = useState(null)
  const [uploadResult, setUploadResult] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [checkingAccess, setCheckingAccess] = useState(true)
  const [accessError, setAccessError] = useState(null)
  const [accessRetryKey, setAccessRetryKey] = useState(0)
  const [intakeWarnings, setIntakeWarnings] = useState([])
  const [inputMode, setInputMode] = useState('upload') // 'upload' | 'live'
  const [liveStreaming, setLiveStreaming] = useState(false)
  const [streamKey, setStreamKey] = useState(0)
  const [liveStatus, setLiveStatus] = useState(null)
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  const fileInputRef = useRef(null)
  const lastSpokenCueSeqRef = useRef(0)
  const lastSpokenAlignmentSeqRef = useRef(0)
  const previousStateRef = useRef(null)

  // Poll the live-status endpoint while the camera is actually streaming —
  // same pattern as RehabPanel, see that component for the full reasoning.
  useEffect(() => {
    if (!liveStreaming) {
      setLiveStatus(null)
      return
    }
    let cancelled = false
    const tick = async () => {
      const data = await getSitToStandLiveStatus(lastSpokenCueSeqRef.current, lastSpokenAlignmentSeqRef.current)
      if (!cancelled && data) setLiveStatus(data)
    }
    tick()
    const id = setInterval(tick, LIVE_STATUS_POLL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [liveStreaming])

  // Rep-quality voice cues ("Good", "Slow down", "Stand up further") — drains
  // cue_pending (everything since our last-seen seq), not just the single
  // "current" cue value. See RehabPanel's identical effect for why: two cues
  // can legitimately fire within one poll interval, and a single-value
  // comparison would silently lose whichever fired first.
  useEffect(() => {
    if (!voiceEnabled || !liveStatus?.active || !liveStatus.cue_pending?.length) return
    for (const { seq, text } of liveStatus.cue_pending) {
      lastSpokenCueSeqRef.current = Math.max(lastSpokenCueSeqRef.current, seq)
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(text))
      }
    }
  }, [liveStatus, voiceEnabled])

  // Camera-alignment cue on its own dedup channel, same reasoning as
  // RehabPanel: kept separate from rep-quality cues so a frequent rep event
  // can't silently overwrite a rarer alignment transition before it's heard.
  useEffect(() => {
    if (!voiceEnabled || !liveStatus?.active || !liveStatus.alignment_cue_pending?.length) return
    for (const { seq, text } of liveStatus.alignment_cue_pending) {
      lastSpokenAlignmentSeqRef.current = Math.max(lastSpokenAlignmentSeqRef.current, seq)
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(text))
      }
    }
  }, [liveStatus, voiceEnabled])

  // Speaks each state's hint text aloud on every real state transition —
  // same pattern as RehabPanel (skips the very first render so it only
  // announces actual transitions, not the initial state on page load).
  useEffect(() => {
    if (previousStateRef.current === null) {
      previousStateRef.current = state
      return
    }
    if (previousStateRef.current === state) return
    previousStateRef.current = state
    if (!voiceEnabled) return
    const hint = STATE_COPY[state]?.hint
    if (hint && typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(hint))
    }
  }, [state, voiceEnabled])

  // Same login + intake gate as RehabPanel, checked against the SAME backend
  // (knee-extension's, port 5050) — both Flask apps share one user database
  // and (as long as REHAB_SECRET_KEY isn't overridden differently per
  // process) one signed session cookie, so there's no separate sit-to-stand
  // login step. Sit-to-stand handles real medical/biomechanical fall-risk
  // data the same way knee extension does, so it gets the same gate, not a
  // lighter one.
  useEffect(() => {
    let cancelled = false
    setCheckingAccess(true)
    setAccessError(null)
    async function checkAccess() {
      try {
        const user = await getCurrentUser()
        if (!user) {
          if (!cancelled) navigate('/login', { state: { redirectTo: '/intake', mode: location.state?.mode } })
          return
        }
        const intake = await getCurrentIntake()
        if (!intake) {
          if (!cancelled) navigate('/intake', { state: location.state })
          return
        }
        const stored = sessionStorage.getItem('fitbuddy-intake-warnings')
        if (!cancelled) {
          setIntakeWarnings(stored ? JSON.parse(stored) : [])
          setCheckingAccess(false)
        }
      } catch (err) {
        if (!cancelled) {
          setAccessError(err.message || 'Could not reach the backend. Is it running?')
          setCheckingAccess(false)
        }
      }
    }
    checkAccess()
    return () => { cancelled = true }
  }, [navigate, location.state, accessRetryKey])

  const canRecord = RECORD_STATES.includes(state)

  if (checkingAccess || accessError) {
    return (
      <div className="dash-grid">
        <div className="dash-card video-panel">
          {accessError ? (
            <>
              <p className="out-feedback-text" style={{ color: '#e06c75' }}>⚠ {accessError}</p>
              <button className="ghost-btn" onClick={() => setAccessRetryKey(k => k + 1)}>
                Retry
              </button>
            </>
          ) : (
            <p className="out-feedback-text">Checking your account and intake status…</p>
          )}
        </div>
      </div>
    )
  }

  function handleFile(file) {
    if (!file || !file.type.startsWith('video/')) return
    setVideoFile(file)
    setVideoSrc(URL.createObjectURL(file))
    setUploadResult(null)
    setError(null)
  }

  function onFileChange(e) {
    handleFile(e.target.files[0])
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  async function handleProcess() {
    if (!videoFile) return
    setProcessing(true)
    setError(null)
    setUploadResult(null)
    try {
      setUploadResult(await uploadSitToStandVideo(videoFile))
    } catch (err) {
      setError(err.message)
    } finally {
      setProcessing(false)
    }
  }

  function handleStartCamera() {
    setStreamKey(k => k + 1)
    lastSpokenCueSeqRef.current = 0
    lastSpokenAlignmentSeqRef.current = 0
    setLiveStreaming(true)
  }

  async function handleStopCamera() {
    setLiveStreaming(false)
    if (!canRecord) return
    try {
      setUploadResult(await getSitToStandClipSummary())
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleAdvance() {
    setError(null)
    try {
      const data = await advanceSitToStandState()
      setState(data.state)
      if (data.state !== 'STATE_ANALYZE') setReport(null)
      setVideoSrc(null)
      setVideoFile(null)
      setUploadResult(null)
      setLiveStreaming(false)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleReport() {
    setError(null)
    try {
      setReport(await getSitToStandReport())
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleReset() {
    setError(null)
    try {
      const data = await resetSitToStandSession()
      setState(data.state)
      setReport(null)
      setVideoSrc(null)
      setVideoFile(null)
      setUploadResult(null)
      setLiveStreaming(false)
    } catch (err) {
      setError(err.message)
    }
  }

  const copy = STATE_COPY[state] ?? STATE_COPY.STATE_SELECT

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {intakeWarnings.length > 0 && (
        <div className="dash-card" style={{ minHeight: 0, gap: 10 }}>
          <div className="panel-label">FROM YOUR INTAKE</div>
          {intakeWarnings.map((w, i) => (
            <div key={i} className="out-item out-item-neutral" style={{ borderLeftColor: INTAKE_WARNING_COLOR[w.level] }}>
              <div className="out-item-dot" style={{ background: INTAKE_WARNING_COLOR[w.level] }} />
              <div>
                <p className="out-item-detail" style={{ color: 'var(--text-primary)' }}>{w.message}</p>
                {w.source && (
                  <a href={w.source} target="_blank" rel="noreferrer" style={{ fontSize: '0.75rem', color: 'var(--blue-light)' }}>
                    Source
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className={`dash-grid ${canRecord ? 'is-recording' : ''}`}>
      <div className="dash-card video-panel">
        <div className="panel-label">SIT-TO-STAND CLIP INPUT</div>

        {canRecord && (
          <div className="mode-tabs" style={{ marginBottom: 0 }}>
            <button
              className={`mode-tab ${inputMode === 'upload' ? 'mode-tab-active' : ''}`}
              onClick={() => { setInputMode('upload'); handleStopCamera() }}
              type="button"
            >
              Upload Clip
            </button>
            <button
              className={`mode-tab ${inputMode === 'live' ? 'mode-tab-active' : ''}`}
              onClick={() => { setInputMode('live'); setVideoSrc(null); setVideoFile(null); setUploadResult(null) }}
              type="button"
            >
              Live Camera
            </button>
          </div>
        )}

        {!canRecord && (
          <div className="upload-zone" style={{ cursor: 'default' }}>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Click "Next" until you reach the recording step to record or upload a clip.
            </p>
          </div>
        )}

        {canRecord && inputMode === 'live' && (
          liveStreaming ? (
            <div className="video-wrapper">
              <img key={streamKey} src={`${SIT_TO_STAND_STREAM_URL}?session=${streamKey}`} alt="Live camera feed with pose overlay" className="video-preview" />
              <LiveStatusHud status={liveStatus} />
              <div className="video-actions">
                <button className="ghost-btn" onClick={handleStopCamera}>Stop Camera</button>
                <button className="ghost-btn" onClick={() => setVoiceEnabled(v => !v)}>
                  {voiceEnabled ? '🔊 Voice On' : '🔇 Voice Off'}
                </button>
                <span className="cta-sub" style={{ paddingLeft: 0 }}>
                  Perform your reps, then click "Stop Camera" and "Next" when done.
                </span>
              </div>
            </div>
          ) : (
            <div className="upload-zone" onClick={handleStartCamera}>
              <p style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>Start Camera</p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                Streams from the camera attached to the computer running the backend
                (not necessarily this device) — side-on view, seated to start.
              </p>
            </div>
          )
        )}

        {canRecord && inputMode === 'upload' && (
          videoSrc ? (
            <div className="video-wrapper">
              <video src={videoSrc} controls className="video-preview" />
              <div className="video-actions">
                <button className="ghost-btn" onClick={() => { setVideoSrc(null); setVideoFile(null); setUploadResult(null) }}>
                  Remove
                </button>
                <button className="cta-btn" onClick={handleProcess} disabled={processing} style={{ padding: '10px 22px', fontSize: '0.9rem' }}>
                  {processing ? 'Processing…' : 'Process Clip'}
                </button>
              </div>
            </div>
          ) : (
            <div
              className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
              onClick={() => fileInputRef.current.click()}
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
            >
              <p style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>Upload Sit-to-Stand Clip</p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                Side-on view, seated to start, standing fully upright at the top of each rep.
              </p>
              <input ref={fileInputRef} type="file" accept="video/*" hidden onChange={onFileChange} />
            </div>
          )
        )}

        {uploadResult && <ClipSummary data={uploadResult} />}
      </div>

      <div className="dash-card output-panel">
        <div className="panel-label">SIT-TO-STAND SESSION</div>

        {error && (
          <div className="output-error">
            <span style={{ color: '#e06c75' }}>⚠ {error}</span>
          </div>
        )}

        <div className="rehab-state-block">
          <p className="block-title">{copy.title}</p>
          <p className="out-feedback-text">{copy.hint}</p>

          <div className="rehab-btn-row">
            <button className="cta-btn" style={{ padding: '10px 22px', fontSize: '0.9rem' }} onClick={handleAdvance} disabled={state === 'STATE_ANALYZE'}>
              Next
            </button>
            <button className="ghost-btn" onClick={handleReport} disabled={state !== 'STATE_ANALYZE'}>
              Get Report
            </button>
            <button className="ghost-btn" onClick={handleReset}>
              Reset
            </button>
          </div>
        </div>

        {report && <SessionReport data={report} />}
      </div>
      </div>
    </div>
  )
}

// Live rep counter + ROM gauge + current cue, same purpose/shape as
// RehabPanel's LiveStatusHud — reuses classifyLiveRom directly since it's
// already generic (angle + [min, max] target_range, both supplied by the
// backend's own /live-status payload).
function LiveStatusHud({ status }) {
  if (!status || !status.active) return null

  if (status.settling) {
    return (
      <div className="live-hud">
        <span className="live-hud-settling">Get positioned — tracking starts in a moment…</span>
      </div>
    )
  }

  const rom = classifyLiveRom(status.angle, status.target_range)
  const misaligned = status.alignment === 'caution' || status.alignment === 'poor'

  return (
    <div className="live-hud" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 10 }}>
      {misaligned && (
        <div className="live-hud-alignment-warning">⚠ Check your camera angle</div>
      )}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 20 }}>
        <div className="live-hud-reps">
          <span className="live-hud-reps-count">{status.rep_count ?? 0}</span>
          <span className="live-hud-reps-label">REPS</span>
        </div>
        <div className="live-hud-gauge">
          <div className="rom-gauge-track">
            <div className="rom-gauge-fill" style={{ width: `${rom.pct * 100}%`, background: rom.color }} />
          </div>
          <span className="rom-gauge-label" style={{ color: rom.color }}>{rom.label}</span>
        </div>
        {status.cue && (
          <span className="live-hud-cue" style={{ color: 'var(--text-primary)' }}>{status.cue}</span>
        )}
      </div>
      {(status.hip_angle != null || status.knee_angle != null) && (
        <div style={{ display: 'flex', gap: 14, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          {status.hip_angle != null && <span>Hip: {Math.round(status.hip_angle)}°</span>}
          {status.knee_angle != null && <span>Knee: {Math.round(status.knee_angle)}°</span>}
        </div>
      )}
    </div>
  )
}

function InsightItem({ label, value, color, note }) {
  return (
    <div className="out-item out-item-neutral" style={{ borderLeftColor: color }}>
      <div className="out-item-dot" style={{ background: color }} />
      <div>
        <p className="out-item-aspect">
          {label}{value !== undefined && <span style={{ color, fontWeight: 700 }}> · {value}</span>}
        </p>
        {note && <p className="out-item-detail">{note}</p>}
      </div>
    </div>
  )
}

// Per-clip feedback shown right after a clip is processed (upload or live
// stop). No left/right split (bilateral movement, one candidate track wins),
// so this is simpler than RehabPanel's ClipSummary — no LSI/limb comparison.
function ClipSummary({ data }) {
  const alignment = classifyCameraAlignment(data.camera_alignment)
  const tracking = classifyTrackingQuality(data.frames_with_pose, data.frames_processed)
  const best = data.reps?.length ? data.reps.reduce((a, b) => (a.standing_angle >= b.standing_angle ? a : b)) : null
  const deficit = best ? classifyStandingDeficit(best.standing_deficit_deg) : null
  const jerky = best ? classifyJerkyMotion(data.reps.some(r => r.is_jerky)) : null
  const fastRep = data.reps?.find(r => r.is_descent_too_fast)
  const pacing = fastRep ? classifyEccentricPacing(fastRep.eccentric_duration_s, true) : null

  const cues = [alignment, deficit, jerky, pacing].map(c => c?.cue).filter(Boolean)

  return (
    <div className="rehab-upload-summary">
      <p className="block-title" style={{ marginBottom: 4 }}>What We Saw</p>
      <p className="out-item-detail" style={{ marginBottom: 10 }}>
        We automatically locked onto whichever side of your body was more clearly
        tracked in the clip (camera angle can flip which side MediaPipe calls
        "{data.detected_landmark_side || data.side}") — this only affects internal labeling, not accuracy.
      </p>

      {cues.length > 0 ? (
        <div className="rehab-coach-callout" style={{ marginBottom: 12 }}>
          <p className="panel-label" style={{ marginBottom: 6 }}>COACH SAYS</p>
          {cues.map((cue, i) => (
            <p key={i} className="out-feedback-text" style={{ fontWeight: 600 }}>
              {cue}
            </p>
          ))}
        </div>
      ) : (
        <p className="out-feedback-text" style={{ fontWeight: 600, color: '#56c596', marginBottom: 12 }}>
          Nice rep — good form on every metric we track.
        </p>
      )}

      {alignment?.note && <InsightItem label={alignment.label} color={alignment.color} note={alignment.note} />}
      {tracking.note && <InsightItem label={tracking.label} color={tracking.color} note={tracking.note} />}
      {deficit && <InsightItem label={deficit.label} value={`${deficit.value}°`} color={deficit.color} note={deficit.note} />}
      {jerky && <InsightItem label={jerky.label} color={jerky.color} note={jerky.note} />}

      {data.reps?.length > 0 && <RepTable reps={data.reps} />}
    </div>
  )
}

function RepTable({ reps }) {
  return (
    <div className="rep-breakdown">
      <p className="block-title" style={{ marginTop: 14, marginBottom: 8 }}>
        Rep-by-Rep ({reps.length} detected)
      </p>
      <div className="rep-table-wrap">
        <table className="rep-table">
          <thead>
            <tr>
              <th>Rep</th>
              <th>Peak Angle</th>
              <th>Hip / Knee</th>
              <th>Deficit</th>
              <th>Speed</th>
              <th>Descent</th>
              <th>Pattern</th>
            </tr>
          </thead>
          <tbody>
            {reps.map(rep => {
              const deficit = classifyStandingDeficit(rep.standing_deficit_deg)
              const pacing = classifyEccentricPacing(rep.eccentric_duration_s, rep.is_descent_too_fast)
              return (
                <tr key={rep.rep_number}>
                  <td>{rep.rep_number}</td>
                  <td>{rep.standing_angle}°</td>
                  <td>{rep.hip_angle_deg ?? '—'}° / {rep.knee_angle_deg ?? '—'}°</td>
                  <td style={{ color: deficit.color }}>{rep.standing_deficit_deg}°</td>
                  <td>{rep.peak_velocity_deg_s}°/s</td>
                  <td style={{ color: pacing?.color ?? 'var(--text-muted)' }}>
                    {pacing ? `${pacing.value}${rep.is_descent_too_fast ? ' (fast)' : ''}` : '—'}
                  </td>
                  <td style={{ color: rep.is_jerky ? '#e06c75' : '#56c596' }}>
                    {rep.is_jerky ? 'Uneven' : 'Smooth'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// Session-level report (POST /advance -> ANALYZE -> GET /report). No LSI —
// sit-to-stand has no per-limb comparison — so this is a single summary
// block, not RehabPanel's two-limb-card layout.
function SessionReport({ data }) {
  const deficit = classifyStandingDeficit(data.standing_deficit_deg)
  const jerky = classifyJerkyMotion(data.is_jerky)

  return (
    <div className="output-content">
      <p className="out-feedback-text" style={{ marginBottom: 16 }}>
        Best rep this session reached <strong>{data.peak_standing_angle_deg}°</strong>,
        {' '}peak speed <strong>{data.peak_velocity_deg_s}°/s</strong>,
        {' '}across <strong>{data.rep_count}</strong> detected rep{data.rep_count === 1 ? '' : 's'}.
      </p>

      <InsightItem label={deficit.label} value={`${deficit.value}°`} color={deficit.color} note={deficit.note} />
      <InsightItem label={jerky.label} color={jerky.color} note={jerky.note} />

      {data.rejected_frame_count > 0 && (
        <p className="out-item-detail" style={{ marginTop: 10, color: 'var(--text-muted)' }}>
          {data.rejected_frame_count} frame{data.rejected_frame_count === 1 ? '' : 's'} were rejected as
          implausible tracking glitches during this session and excluded from the numbers above —
          not an error, just the pipeline protecting itself against bad camera data.
        </p>
      )}

      {data.reps?.length > 0 && <RepTable reps={data.reps} />}
    </div>
  )
}
