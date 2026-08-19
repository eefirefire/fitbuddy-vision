import { useState, useRef, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { uploadRehabVideo, advanceRehabState, getRehabReport, resetRehabSession, getLiveClipSummary, getLiveStatus, REHAB_STREAM_URL } from '../rehabService'
import { getCurrentUser, getCurrentIntake } from '../authService'
import {
  classifyTrackingQuality, classifyExtensionDeficit,
  classifyMovementSpeed, classifyMuscleActivation, classifyLSI, classifyRepTrend,
  classifyCameraAlignment, classifyTrunkCompliance, classifyEccentricPacing, classifyLiveRom,
} from '../rehabInterpret'

// Live-status is polled rather than pushed (no websocket in this stack — see
// Scripts/rehab_knee_extension.py's live_status()), so this just needs to be
// fast enough to feel live without hammering the backend. 400ms keeps the
// rep counter and gauge feeling responsive without meaningfully loading a
// Flask dev server running alongside the MJPEG stream.
const LIVE_STATUS_POLL_MS = 400

const INTAKE_WARNING_COLOR = { red: '#e06c75', amber: '#e5a14a' }

const STATE_COPY = {
  STATE_SELECT_LEFT: { title: 'Select Left Leg', hint: 'Sit side-on to the camera, left leg facing the lens. Click "Next" when ready.' },
  STATE_RECORD_LEFT: { title: 'Recording: Left Leg', hint: 'Perform your seated knee extension reps now.' },
  STATE_PROMPT_SWITCH: { title: 'Switch Sides', hint: 'Turn around so your right leg faces the camera, then click "Next".' },
  STATE_SELECT_RIGHT: { title: 'Select Right Leg', hint: 'Sit side-on to the camera, right leg facing the lens. Click "Next" when ready.' },
  STATE_RECORD_RIGHT: { title: 'Recording: Right Leg', hint: 'Perform your seated knee extension reps now.' },
  STATE_ANALYZE: { title: 'Analysis Ready', hint: 'Click "Get Report" to see the Limb Symmetry Index.' },
}

const RECORD_STATES = ['STATE_RECORD_LEFT', 'STATE_RECORD_RIGHT']

export default function RehabPanel() {
  const navigate = useNavigate()
  const location = useLocation()
  const [state, setState] = useState('STATE_SELECT_LEFT')
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
  // drives the rep counter, ROM gauge, and voice cues below. Stops cleanly
  // whenever streaming ends so it never polls in the background.
  useEffect(() => {
    if (!liveStreaming) {
      setLiveStatus(null)
      return
    }
    let cancelled = false
    const tick = async () => {
      const data = await getLiveStatus()
      if (!cancelled && data) setLiveStatus(data)
    }
    tick()
    const id = setInterval(tick, LIVE_STATUS_POLL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [liveStreaming])

  // Voice feedback — speaks only on a genuine state change (a new cue_seq
  // from the backend), never on every poll tick, so it doesn't repeat the
  // same phrase 2-3x a second. Web Speech API only: offline, no external
  // service, matches the "works without reliable venue wifi" constraint.
  useEffect(() => {
    if (!voiceEnabled || !liveStatus?.active || !liveStatus.cue) return
    if (liveStatus.cue_seq === lastSpokenCueSeqRef.current) return
    lastSpokenCueSeqRef.current = liveStatus.cue_seq
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      // Deliberately NOT calling speechSynthesis.cancel() here. Each rep can
      // legitimately fire two distinct cues moments apart — the peak-extension
      // cue ("Good"/"Extend further") the instant you reach full extension,
      // then the descent-timing cue ("Slow down") a beat later once the leg
      // finishes lowering. cancel() was killing whichever utterance was still
      // playing when the second cue arrived, so in practice "Good" almost
      // never got heard — only the last cue of each rep did. The Web Speech
      // API queues speak() calls by default, so both play back to back
      // instead; the cue_seq dedup above already guarantees this only fires
      // on genuinely new cues, so the queue never grows unbounded.
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(liveStatus.cue))
    }
  }, [liveStatus, voiceEnabled])

  // Camera-alignment cue speaks on its OWN dedup channel (alignment_cue_seq,
  // not cue_seq) — the backend keeps this entirely separate from the
  // rep-quality cue above. They used to share one "current cue" slot, which
  // meant a rep cue firing shortly after an alignment cue would silently
  // overwrite it before this component ever saw it: rep events happen more
  // often than alignment transitions, so "Slow down"/"Good" would almost
  // always win the race and "Check your camera angle" would never actually
  // get spoken, even though it fired correctly on the backend. Independent
  // channels mean both can be heard.
  useEffect(() => {
    if (!voiceEnabled || !liveStatus?.active || !liveStatus.alignment_cue) return
    if (liveStatus.alignment_cue_seq === lastSpokenAlignmentSeqRef.current) return
    lastSpokenAlignmentSeqRef.current = liveStatus.alignment_cue_seq
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(liveStatus.alignment_cue))
    }
  }, [liveStatus, voiceEnabled])

  // Speaks each state's hint text (STATE_COPY[state].hint) aloud on every
  // real state transition — most useful for STATE_PROMPT_SWITCH ("turn
  // around so your right leg faces the camera"), so there's no need to walk
  // back to the laptop just to find out what to do next. Reuses the same
  // Web Speech API approach as the rep/alignment cues above. Skips the very
  // first render (previousStateRef starts null) so it only announces actual
  // transitions, not the initial "Select Left Leg" the instant the page
  // loads — and doesn't re-fire just because voiceEnabled was toggled while
  // sitting on the same state.
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

  // Rehab mode handles real medical/biomechanical data, so it's gated behind
  // login + a completed pre-exercise intake — checked against the backend
  // (not just local state) since that's the actual source of truth enforced
  // by @login_required on every /api/rehab/* route.
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
        // Most commonly: the backend (localhost:5050) isn't running yet, so
        // fetch() throws instead of resolving with a normal HTTP status.
        // Without this catch, the component hung on "Checking..." forever,
        // even after the backend came up, since the effect had already run
        // and failed silently.
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
      setUploadResult(await uploadRehabVideo(videoFile))
    } catch (err) {
      setError(err.message)
    } finally {
      setProcessing(false)
    }
  }

  function handleStartCamera() {
    setStreamKey(k => k + 1) // cache-busts the <img> src so each Start gets a fresh stream connection
    lastSpokenCueSeqRef.current = 0 // fresh recording — don't carry over a stale cue_seq from a previous session
    lastSpokenAlignmentSeqRef.current = 0
    setLiveStreaming(true)
  }

  async function handleStopCamera() {
    // Removing the <img> closes the underlying HTTP connection, releasing
    // the webcam. Then immediately fetch the clip summary so the user gets
    // the same feedback card as after an upload.
    setLiveStreaming(false)
    if (!canRecord) return
    try {
      setUploadResult(await getLiveClipSummary())
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleAdvance() {
    setError(null)
    try {
      const data = await advanceRehabState()
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
      setReport(await getRehabReport())
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleReset() {
    setError(null)
    try {
      const data = await resetRehabSession()
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

  const copy = STATE_COPY[state] ?? STATE_COPY.STATE_SELECT_LEFT

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
      {/* Left: video input panel — upload a clip, or stream straight from the camera */}
      <div className="dash-card video-panel">
        <div className="panel-label">REHAB CLIP INPUT</div>

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
              Click "Next" until you reach a recording step to record or upload a clip.
            </p>
          </div>
        )}

        {canRecord && inputMode === 'live' && (
          liveStreaming ? (
            <div className="video-wrapper">
              <img key={streamKey} src={`${REHAB_STREAM_URL}?session=${streamKey}`} alt="Live camera feed with pose overlay" className="video-preview" />
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
                (not necessarily this device) — side-on view, knee starting bent.
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
              <p style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>Upload Rehab Clip</p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                Side-on view, knee starting bent and extending to straight.
              </p>
              <input ref={fileInputRef} type="file" accept="video/*" hidden onChange={onFileChange} />
            </div>
          )
        )}

        {uploadResult && <ClipSummary data={uploadResult} />}
      </div>

      {/* Right: state machine + report panel */}
      <div className="dash-card output-panel">
        <div className="panel-label">REHAB SESSION</div>

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

        {report && <RehabReport data={report} />}
      </div>
      </div>
    </div>
  )
}

// Live rep counter + ROM gauge + current cue, shown under the camera feed
// while a live recording is in progress. The video overlay (drawn
// server-side) already shows the same rep count/cue burned into the pixels
// as a stage-safe fallback with zero network dependency — this is the same
// data surfaced as real DOM text so it's legible at a glance and can drive
// the voice cues, which the video overlay can't.
function LiveStatusHud({ status }) {
  if (!status || !status.active) return null

  // Backend holds off on rep-tracking/cues for a few seconds after the
  // camera starts, so ordinary setup movement (sitting down, adjusting the
  // camera) doesn't get mistaken for reps or trigger spurious cues — see
  // RehabSession.is_settling in the backend. Show a plain "get ready"
  // message instead of a counter/gauge that isn't tracking yet.
  if (status.settling) {
    return (
      <div className="live-hud">
        <span className="live-hud-settling">Get positioned — tracking starts in a moment…</span>
      </div>
    )
  }

  const rom = classifyLiveRom(status.angle, status.target_range)
  // Shown PERSISTENTLY whenever alignment is currently caution/poor (the
  // "alignment" field always reflects the current recent-window reading),
  // not just at the moment the one-shot alignment_cue transition fires —
  // matches the video overlay's behavior so it stays visible for as long as
  // the camera is actually off-angle, rather than flashing once.
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
    </div>
  )
}

// A single colored insight row — mirrors the squat dashboard's good/warn item style.
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

// Per-clip feedback shown right after a leg's video is processed.
function ClipSummary({ data }) {
  const alignment = classifyCameraAlignment(data.camera_alignment)
  const trunk = classifyTrunkCompliance(data.trunk_compliance)
  const tracking = classifyTrackingQuality(data.frames_with_pose, data.frames_processed)
  const deficit = classifyExtensionDeficit(data.extension_deficit_deg)
  const speed = classifyMovementSpeed(data.peak_velocity_deg_s)
  const activation = classifyMuscleActivation(data.is_inhibited)
  const trend = data.reps?.length >= 2 ? classifyRepTrend(data.reps) : null
  // eccentric_duration_s/is_descent_too_fast live per-rep, not as a clip-level
  // summary field — surface one pacing cue at this level if ANY rep dropped
  // instead of lowering under control; the full per-rep timing is still in
  // RepTable below.
  const fastRep = data.reps?.find(r => r.is_descent_too_fast)
  const pacing = fastRep ? classifyEccentricPacing(fastRep.eccentric_duration_s, true) : null

  // Lead with plain "what to do about it" cues — the cards below still carry
  // the full numbers and explanations for anyone who wants the detail.
  const cues = [alignment, trunk, deficit, speed, activation, pacing, trend]
    .map(c => c?.cue)
    .filter(Boolean)

  return (
    <div className="rehab-upload-summary">
      <p className="block-title" style={{ marginBottom: 4 }}>
        {data.side === 'left' ? 'Left' : 'Right'} Leg — What We Saw
      </p>
      <p className="out-item-detail" style={{ marginBottom: 10 }}>
        We automatically locked onto whichever leg was actually moving in the clip
        (camera angle can flip which side MediaPipe calls "{data.detected_landmark_side}") —
        this only affects internal labeling, not accuracy.
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
      {trunk?.note && <InsightItem label={trunk.label} color={trunk.color} note={trunk.note} />}
      {tracking.note && <InsightItem label={tracking.label} color={tracking.color} note={tracking.note} />}
      <InsightItem label={deficit.label} value={`${deficit.value}°`} color={deficit.color} note={deficit.note} />
      <InsightItem label={speed.label} value={`${speed.value}°/s`} color={speed.color} note={speed.note} />
      <InsightItem label={activation.label} color={activation.color} note={activation.note} />

      {data.reps?.length > 0 && <RepBreakdown reps={data.reps} />}
    </div>
  )
}

// Shows every individual rep detected in the clip, not just the best one —
// a "best rep" summary hides whether the patient is staying consistent or
// fatiguing/loosening up across the set, which matters clinically. Combines
// an automatic trend read, a visual chart, and the exact numbers in a table
// — the chart makes the pattern obvious at a glance, the table gives the
// precise values a clinician would want.
function RepBreakdown({ reps }) {
  const trend = classifyRepTrend(reps)

  return (
    <div className="rep-breakdown">
      <p className="block-title" style={{ marginTop: 14, marginBottom: 8 }}>
        Rep-by-Rep ({reps.length} detected)
      </p>

      {reps.length >= 2 && (
        <InsightItem label={trend.label} color={trend.color} note={trend.note} />
      )}

      <div className="rep-chart-wrap">
        <RepChart reps={reps} />
      </div>

      <RepTable reps={reps} />
    </div>
  )
}

// Bar height = how close to full (180°) extension that rep reached. Bar
// color reuses the same deficit severity scale shown elsewhere, so the
// chart and the rest of the page agree on what "bad" looks like.
function RepChart({ reps }) {
  const DOMAIN_MIN = 90
  const DOMAIN_MAX = 180
  const width = Math.max(reps.length * 64, 180)
  const height = 130
  const padTop = 14
  const padBottom = 22
  const chartHeight = height - padTop - padBottom
  const slot = width / reps.length
  const barWidth = slot * 0.5

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img" aria-label="Peak extension angle per rep">
      <line x1={0} y1={padTop} x2={width} y2={padTop} stroke="var(--border)" strokeDasharray="4 3" />
      <text x={2} y={padTop - 4} fontSize="9" fill="var(--text-muted)">180°</text>

      {reps.map((rep, i) => {
        const cx = slot * i + slot / 2
        const pct = Math.max(0, Math.min(1, (rep.peak_angle - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)))
        const barHeight = Math.max(2, pct * chartHeight)
        const color = classifyExtensionDeficit(rep.extension_deficit_deg).color
        return (
          <g key={rep.rep_number}>
            <rect
              x={cx - barWidth / 2}
              y={padTop + chartHeight - barHeight}
              width={barWidth}
              height={barHeight}
              rx={3}
              fill={color}
              opacity={0.85}
            />
            <text x={cx} y={height - 6} fontSize="9" fill="var(--text-muted)" textAnchor="middle">
              #{rep.rep_number}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function RepTable({ reps }) {
  return (
    <div className="rep-table-wrap">
      <table className="rep-table">
        <thead>
          <tr>
            <th>Rep</th>
            <th>Peak Angle</th>
            <th>Deficit</th>
            <th>Speed</th>
            <th>Descent</th>
            <th>Pattern</th>
          </tr>
        </thead>
        <tbody>
          {reps.map(rep => {
            const deficit = classifyExtensionDeficit(rep.extension_deficit_deg)
            const pacing = classifyEccentricPacing(rep.eccentric_duration_s, rep.is_descent_too_fast)
            return (
              <tr key={rep.rep_number}>
                <td>{rep.rep_number}</td>
                <td>{rep.peak_angle}°</td>
                <td style={{ color: deficit.color }}>{rep.extension_deficit_deg}°</td>
                <td>{rep.peak_velocity_deg_s}°/s</td>
                <td style={{ color: pacing?.color ?? 'var(--text-muted)' }}>
                  {pacing ? `${pacing.value}${rep.is_descent_too_fast ? ' (fast)' : ''}` : '—'}
                </td>
                <td style={{ color: rep.is_inhibited ? '#e06c75' : '#56c596' }}>
                  {rep.is_inhibited ? 'Stuttered' : 'Smooth'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function RehabReport({ data }) {
  const velocityLsi = classifyLSI(data.velocity_lsi_pct)
  const angleLsi = classifyLSI(data.angle_lsi_pct)
  const weakerLeg = data.left.peak_velocity_deg_s <= data.right.peak_velocity_deg_s ? 'Left' : 'Right'

  return (
    <div className="output-content">
      <p className="out-feedback-text" style={{ marginBottom: 16 }}>
        Comparing your two legs: your <strong>{weakerLeg} leg</strong> is currently the weaker side.
        Here's how closely they're matched.
      </p>

      <div className="out-rating-block">
        <div className="out-rating-row">
          <span className="block-title">Strength Symmetry (speed)</span>
          <span className="out-rating-val" style={{ color: velocityLsi.color }}>
            {data.velocity_lsi_pct}<small> %</small>
          </span>
        </div>
        <div className="score-bar-bg">
          <div className="score-bar-fill" style={{ width: `${Math.min(data.velocity_lsi_pct, 100)}%`, background: velocityLsi.color }} />
        </div>
        <p className="out-item-detail" style={{ marginTop: 6 }}>{velocityLsi.note}</p>
      </div>

      <div className="out-rating-block">
        <div className="out-rating-row">
          <span className="block-title">Range-of-Motion Symmetry</span>
          <span className="out-rating-val" style={{ color: angleLsi.color }}>
            {data.angle_lsi_pct}<small> %</small>
          </span>
        </div>
        <div className="score-bar-bg">
          <div className="score-bar-fill" style={{ width: `${Math.min(data.angle_lsi_pct, 100)}%`, background: angleLsi.color }} />
        </div>
        <p className="out-item-detail" style={{ marginTop: 6 }}>{angleLsi.note}</p>
      </div>

      <div className="limb-compare-grid">
        <LimbCard title="Left Leg" limb={data.left} />
        <LimbCard title="Right Leg" limb={data.right} />
      </div>
    </div>
  )
}

function LimbCard({ title, limb }) {
  const deficit = classifyExtensionDeficit(limb.extension_deficit_deg)
  const speed = classifyMovementSpeed(limb.peak_velocity_deg_s)
  const activation = classifyMuscleActivation(limb.is_inhibited)

  return (
    <div>
      <p className="block-title" style={{ marginBottom: 8 }}>{title}</p>
      <p className="out-item-detail" style={{ color: deficit.color }}>{deficit.label} ({deficit.value}°)</p>
      <p className="out-item-detail" style={{ color: speed.color }}>{speed.label} ({speed.value}°/s)</p>
      <p className="out-item-detail" style={{ color: activation.color }}>{activation.label}</p>
      {limb.reps?.length > 0 && <RepBreakdown reps={limb.reps} />}
    </div>
  )
}
