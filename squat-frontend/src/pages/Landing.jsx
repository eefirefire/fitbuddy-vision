import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

const MODES = [
  {
    id: 'ergonomic',
    icon: '💼',
    tabLabel: 'Ergonomic',
    headline: ['Fix Your', 'Desk Posture.'],
    ctaNoun: 'Posture',
    focus: 'Focus: Workspace',
    points: ['Shoulder slouch', 'Forward head tilt', 'Spinal alignment'],
  },
  {
    id: 'exercise',
    icon: '🏆',
    tabLabel: 'Exercise',
    headline: ['Perfect Your', 'Squat Form.'],
    ctaNoun: 'Squat',
    focus: 'Focus: Fitness Form',
    points: ['Squat depth checks', 'Rep lockouts', 'Descent metrics'],
  },
  {
    id: 'rehab',
    icon: '🏥',
    tabLabel: 'Rehab',
    headline: ['Track Your', 'Rehab Progress.'],
    ctaNoun: 'Rehab Session',
    focus: 'Focus: Medical Rehab',
    points: ['Joint symmetry', 'Range of motion', 'Smooth velocity'],
  },
  {
    id: 'sit-to-stand',
    icon: '🪑',
    tabLabel: 'Sit-to-Stand',
    headline: ['Track Your', 'Sit-to-Stand.'],
    ctaNoun: 'Sit-to-Stand Session',
    focus: 'Focus: Fall-Risk Screening',
    points: ['Rep count', 'Standing deficit', 'Rise smoothness'],
  },
]

function SquatIllustration() {
  return (
    <svg viewBox="0 0 380 320" width="100%" height="100%" style={{ maxHeight: 320 }} xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
          <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <ellipse cx="200" cy="160" rx="145" ry="140" fill="none" stroke="rgba(126,184,212,0.15)" strokeWidth="1" strokeDasharray="6 4"/>
      <ellipse cx="200" cy="160" rx="115" ry="110" fill="none" stroke="rgba(126,184,212,0.1)" strokeWidth="1" strokeDasharray="4 6"/>
      <circle cx="200" cy="62" r="16" fill="none" stroke="#7eb8d4" strokeWidth="2" filter="url(#glow)"/>
      <line x1="200" y1="78" x2="196" y2="130" stroke="#7eb8d4" strokeWidth="2"/>
      <line x1="196" y1="95" x2="165" y2="115" stroke="#7eb8d4" strokeWidth="2"/>
      <line x1="196" y1="95" x2="228" y2="115" stroke="#7eb8d4" strokeWidth="2"/>
      <line x1="165" y1="115" x2="150" y2="138" stroke="#7eb8d4" strokeWidth="2"/>
      <line x1="228" y1="115" x2="243" y2="138" stroke="#7eb8d4" strokeWidth="2"/>
      <line x1="196" y1="130" x2="175" y2="128" stroke="#7eb8d4" strokeWidth="2"/>
      <line x1="196" y1="130" x2="218" y2="128" stroke="#7eb8d4" strokeWidth="2"/>
      <line x1="175" y1="128" x2="155" y2="185" stroke="#7eb8d4" strokeWidth="2.5"/>
      <line x1="218" y1="128" x2="238" y2="185" stroke="#7eb8d4" strokeWidth="2.5"/>
      <line x1="155" y1="185" x2="148" y2="240" stroke="#7eb8d4" strokeWidth="2.5"/>
      <line x1="238" y1="185" x2="246" y2="240" stroke="#7eb8d4" strokeWidth="2.5"/>
      <line x1="148" y1="240" x2="132" y2="242" stroke="#7eb8d4" strokeWidth="2"/>
      <line x1="246" y1="240" x2="262" y2="242" stroke="#7eb8d4" strokeWidth="2"/>
      {[
        [200,62],[196,95],[165,115],[228,115],[150,138],[243,138],
        [196,130],[175,128],[218,128],[155,185],[238,185],[148,240],[246,240]
      ].map(([x,y],i) => (
        <circle key={i} cx={x} cy={y} r="4" fill="#7eb8d4" filter="url(#glow)"/>
      ))}
      <path d="M 148 220 A 20 20 0 0 1 168 205" fill="none" stroke="#4a6fa5" strokeWidth="1.5" strokeDasharray="3 2"/>
      <text x="128" y="215" fill="#7eb8d4" fontSize="10" fontFamily="monospace">92°</text>
      <path d="M 180 148 A 18 18 0 0 1 196 158" fill="none" stroke="#4a6fa5" strokeWidth="1.5" strokeDasharray="3 2"/>
      <text x="160" y="168" fill="#7eb8d4" fontSize="10" fontFamily="monospace">78°</text>
      <rect x="270" y="68" width="96" height="28" rx="6" fill="#1a2240" stroke="#4a6fa5" strokeWidth="1"/>
      <circle cx="280" cy="82" r="3" fill="#7eb8d4"/>
      <text x="288" y="87" fill="#7eb8d4" fontSize="9.5" fontFamily="monospace" fontWeight="600">LIVE ANALYSIS</text>
      <rect x="12" y="190" width="88" height="28" rx="6" fill="#1a2240" stroke="#4a6fa5" strokeWidth="1"/>
      <circle cx="22" cy="204" r="3" fill="#7eb8d4"/>
      <text x="30" y="209" fill="#7eb8d4" fontSize="9.5" fontFamily="monospace" fontWeight="600">FORM SCORE</text>
      <line x1="270" y1="82" x2="246" y2="100" stroke="#2d3f6b" strokeWidth="1" strokeDasharray="3 3"/>
      <line x1="100" y1="204" x2="148" y2="200" stroke="#2d3f6b" strokeWidth="1" strokeDasharray="3 3"/>
      <rect x="272" y="105" width="96" height="52" rx="6" fill="#161820" stroke="#2d3f6b" strokeWidth="1"/>
      <text x="320" y="122" fill="#8a9bb5" fontSize="8" textAnchor="middle" fontFamily="monospace">DEPTH</text>
      <rect x="280" y="126" width="80" height="6" rx="3" fill="#1a2240"/>
      <rect x="280" y="126" width="64" height="6" rx="3" fill="#4a6fa5"/>
      <text x="320" y="148" fill="#8a9bb5" fontSize="8" textAnchor="middle" fontFamily="monospace">BALANCE</text>
      <rect x="280" y="139" width="80" height="6" rx="3" fill="#1a2240"/>
      <rect x="280" y="139" width="72" height="6" rx="3" fill="#7eb8d4"/>
    </svg>
  )
}

const MODE_STORAGE_KEY = 'fitbuddy-selected-mode'

export default function Landing() {
  const navigate = useNavigate()
  // Landing remounts fresh every time you navigate back to it (e.g. browser
  // back button from /dashboard), so plain useState would always reset to
  // the default. Reading/writing localStorage makes the choice survive that.
  const [selectedMode, setSelectedMode] = useState(
    () => localStorage.getItem(MODE_STORAGE_KEY) || 'exercise'
  )
  const mode = MODES.find(m => m.id === selectedMode)

  function selectMode(id) {
    setSelectedMode(id)
    localStorage.setItem(MODE_STORAGE_KEY, id)
  }

  function goToDashboard() {
    navigate('/dashboard', { state: { mode: selectedMode } })
  }

  return (
    <div className="grid-bg">
      <nav className="top-nav">
        <span className="nav-brand">FitBuddy AI</span>
        <div className="nav-links">
          [<a href="#home">Home</a>
          <span>|</span>
          <a href="#features">Features</a>
          <span>|</span>
          <a href="#contact">Contact</a>]
        </div>
      </nav>

      <div className="hero-card">
        <div className="hero-left">
          <div className="mode-tabs">
            {MODES.map(m => (
              <button
                key={m.id}
                className={`mode-tab ${selectedMode === m.id ? 'mode-tab-active' : ''}`}
                onClick={() => selectMode(m.id)}
              >
                <span className="mode-tab-icon">{m.icon}</span>{m.tabLabel}
              </button>
            ))}
          </div>

          <h1 className="hero-title">
            {mode.headline[0]}<br />{mode.headline[1]}
          </h1>

          <div className="mode-focus-box">
            <p className="mode-focus-title">{mode.focus}</p>
            <div className="mode-focus-pills">
              {mode.points.map(p => <span key={p} className="mode-focus-pill">{p}</span>)}
            </div>
          </div>

          <div className="cta-group">
            <button className="cta-btn" onClick={goToDashboard}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8" fill="currentColor"/>
              </svg>
              Analyse My {mode.ctaNoun}
            </button>
            <span className="cta-sub">Upload a video and get instant feedback.</span>
          </div>
        </div>

        <div className="hero-right">
          <SquatIllustration />
        </div>

        <div className="card-footer">
          <a href="#privacy">Privacy Policy</a>
          <span>|</span>
          <a href="#terms">Terms of Service</a>
        </div>
      </div>

      <div className="diamond"/>
    </div>
  )
}
