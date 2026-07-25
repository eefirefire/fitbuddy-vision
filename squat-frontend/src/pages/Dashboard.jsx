import { useState, useRef } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { analyzeLift } from '../analyzeService'
import RehabPanel from './RehabPanel'

const MODE_LABELS = {
  ergonomic: '💼 Ergonomic Mode',
  exercise: '🏆 Exercise Mode',
  rehab: '🏥 Rehab Mode',
}

export default function Dashboard() {
  const navigate = useNavigate()
  const location = useLocation()
  const fileInputRef = useRef(null)
  // Falls back to the persisted choice (e.g. after a hard refresh on this
  // page, location.state is gone) before defaulting to 'exercise'.
  const mode = location.state?.mode ?? localStorage.getItem('fitbuddy-selected-mode') ?? 'exercise'

  const [videoSrc, setVideoSrc] = useState(null)
  const [videoFile, setVideoFile] = useState(null)
  const [sam3Enabled, setSam3Enabled] = useState(false)
  const [aiOutput, setAiOutput] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  function handleFile(file) {
    if (!file || !file.type.startsWith('video/')) return
    setVideoFile(file)
    setVideoSrc(URL.createObjectURL(file))
    setAiOutput(null)
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

  async function runAnalysis() {
    if (!videoFile) return
    setLoading(true)
    setError(null)
    setAiOutput(null)

    try {
      const data = await analyzeLift(videoFile, sam3Enabled)
      setAiOutput(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid-bg" style={{ minHeight: '100vh' }}>
      {/* Navbar */}
      <nav className="top-nav">
        <span className="nav-brand" style={{ cursor: 'pointer' }} onClick={() => navigate('/')}>
          ← FitBuddy AI
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <span className="mode-badge">{MODE_LABELS[mode]}</span>
          {/* SAM3 Toggle */}
          <label className="sam3-toggle">
            <span className="toggle-label">SAM3</span>
            <div
              className={`toggle-track ${sam3Enabled ? 'active' : ''}`}
              onClick={() => setSam3Enabled(v => !v)}
            >
              <div className="toggle-thumb"/>
            </div>
            <span className="toggle-state">{sam3Enabled ? 'ON' : 'OFF'}</span>
          </label>
        </div>
      </nav>

      {/* Main grid */}
      {mode === 'rehab' ? <RehabPanel /> : (
      <div className="dash-grid">
        {/* Left: video panel */}
        <div className="dash-card video-panel">
          <div className="panel-label">VIDEO INPUT</div>

          {videoSrc ? (
            <div className="video-wrapper">
              <video
                src={videoSrc}
                controls
                className="video-preview"
              />
              <div className="video-actions">
                <button className="ghost-btn" onClick={() => { setVideoSrc(null); setVideoFile(null); setAiOutput(null) }}>
                  Remove
                </button>
                <button className="cta-btn" onClick={runAnalysis} disabled={loading} style={{ padding: '10px 22px', fontSize: '0.9rem' }}>
                  {loading
                    ? <><Spinner /> Analysing…</>
                    : <>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/>
                        </svg>
                        Run Analysis
                      </>
                  }
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
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: 'var(--blue-mid)', marginBottom: 12 }}>
                <polyline points="16 16 12 12 8 16"/>
                <line x1="12" y1="12" x2="12" y2="21"/>
                <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
              </svg>
              <p style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>Upload Video</p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>Drag & drop or click to browse</p>
              <input ref={fileInputRef} type="file" accept="video/*" hidden onChange={onFileChange}/>
            </div>
          )}
        </div>

        {/* Right: AI output panel */}
        <div className="dash-card output-panel">
          <div className="panel-label">AI OUTPUT</div>

          {error && (
            <div className="output-error">
              <span style={{ color: '#e06c75' }}>⚠ {error}</span>
            </div>
          )}

          {loading && (
            <div className="output-loading">
              <Spinner large />
              <p style={{ color: 'var(--text-muted)', marginTop: 16 }}>Analysing your squat…</p>
            </div>
          )}

          {!loading && !error && !aiOutput && (
            <div className="output-empty">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: 'var(--blue-navy)', marginBottom: 10 }}>
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                AI output will appear here after analysis.
              </p>
            </div>
          )}

          {!loading && aiOutput && <OutputDisplay data={aiOutput} />}
        </div>
      </div>
      )}
    </div>
  )
}

const RATING_COLOR = (r) => {
  if (r >= 8) return '#56c596'
  if (r >= 6) return '#7eb8d4'
  if (r >= 4) return '#e5a14a'
  return '#e06c75'
}

const LOAD_LABELS = { light: 1, moderate: 2, heavy: 3, maximal: 4 }

function OutputDisplay({ data }) {
  if (data.parsing_error) {
    return <pre className="raw-output">{data.raw_text}</pre>
  }

  const rating = data.overall_rating ?? null
  const ratingPct = rating !== null ? (rating / 10) * 100 : null
  const ratingColor = rating !== null ? RATING_COLOR(rating) : 'var(--blue-light)'
  const loadLevel = LOAD_LABELS[data.estimated_load] ?? 0

  return (
    <div className="output-content">

      {/* Header row: exercise + rep count */}
      <div className="out-header">
        <div>
          <p className="out-exercise">{data.exercise?.replace(/_/g, ' ') ?? 'Squat'}</p>
        </div>
        <div className="out-reps">
          <span className="out-reps-num">{data.rep_count ?? '—'}</span>
          <span className="out-reps-label">reps</span>
        </div>
      </div>

      {/* Overall rating bar */}
      {rating !== null && (
        <div className="out-rating-block">
          <div className="out-rating-row">
            <span className="block-title">Overall Rating</span>
            <span className="out-rating-val" style={{ color: ratingColor }}>
              {rating}<small> / 10</small>
            </span>
          </div>
          <div className="score-bar-bg">
            <div className="score-bar-fill" style={{ width: `${ratingPct}%`, background: ratingColor }}/>
          </div>
        </div>
      )}

      {/* Estimated load pills */}
      {data.estimated_load && (
        <div className="out-load-row">
          <span className="block-title" style={{ marginRight: 10 }}>Load</span>
          {['light', 'moderate', 'heavy', 'maximal'].map((l, i) => (
            <span key={l} className={`load-pill ${i < loadLevel ? 'load-pill-on' : ''}`}>{l}</span>
          ))}
        </div>
      )}

      {/* Good points */}
      {data.good?.length > 0 && (
        <div className="out-section">
          <p className="block-title good-title">What's Working</p>
          {data.good.map((g, i) => (
            <div key={i} className="out-item out-item-good">
              <div className="out-item-dot good-dot"/>
              <div>
                <p className="out-item-aspect">{g.aspect}</p>
                <p className="out-item-detail">{g.detail}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Needs improvement */}
      {data.needs_improvement?.length > 0 && (
        <div className="out-section">
          <p className="block-title warn-title">Needs Improvement</p>
          {data.needs_improvement.map((n, i) => (
            <div key={i} className="out-item out-item-warn">
              <div className="out-item-dot warn-dot"/>
              <div>
                <p className="out-item-aspect">{n.aspect}</p>
                <p className="out-item-detail">{n.detail}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Coaching feedback */}
      {data.feedback && (
        <div className="out-feedback">
          <p className="block-title">Coaching Summary</p>
          <p className="out-feedback-text">{data.feedback}</p>
        </div>
      )}

    </div>
  )
}

function Spinner({ large }) {
  const s = large ? 32 : 16
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
      style={{ animation: 'spin 0.8s linear infinite', color: 'var(--blue-light)' }}>
      <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round"/>
    </svg>
  )
}
