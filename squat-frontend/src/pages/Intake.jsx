import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { submitIntake } from '../authService'

const SURGERY_RECENCY_OPTIONS = ['none', '0-2 weeks', '2-4 weeks', '1-3 months', '3-6 months', '6+ months']
const DIAGNOSIS_OPTIONS = ['none', 'ACL reconstruction', 'Meniscus repair', 'Total knee replacement', 'General knee pain', 'Other']

const WARNING_COLOR = { red: '#e06c75', amber: '#e5a14a' }

export default function Intake() {
  const navigate = useNavigate()
  const location = useLocation()
  const [age, setAge] = useState('')
  const [surgeryRecency, setSurgeryRecency] = useState('none')
  const [diagnosis, setDiagnosis] = useState('none')
  const [painLevel, setPainLevel] = useState(0)
  const [swelling, setSwelling] = useState(false)
  const [clearedByDoctor, setClearedByDoctor] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      const data = await submitIntake({
        age: Number(age),
        surgery_recency: surgeryRecency,
        diagnosis,
        pain_level: Number(painLevel),
        swelling,
        cleared_by_doctor: clearedByDoctor,
      })
      setResult(data)
      // RehabPanel shows these as a persistent banner without re-deriving
      // the warning logic client-side — rehab_auth.py stays the single
      // source of truth for the thresholds.
      sessionStorage.setItem('fitbuddy-intake-warnings', JSON.stringify(data.warnings))
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  function handleContinue() {
    navigate('/dashboard', { state: location.state })
  }

  return (
    <div className="grid-bg">
      <nav className="top-nav">
        <span className="nav-brand">FitBuddy AI</span>
      </nav>

      <div className="hero-card" style={{ gridTemplateColumns: '1fr', maxWidth: 560, margin: '0 auto' }}>
        <div className="hero-left">
          <h1 className="hero-title" style={{ fontSize: '1.8rem' }}>Before You Start</h1>
          <p className="out-feedback-text">
            A few quick questions so the warning system can account for factors that affect what's safe for you
            specifically — this isn't a diagnosis, and doesn't replace your doctor or physical therapist.
          </p>

          {!result && (
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <label className="intake-field">
                <span>Age</span>
                <input
                  className="rehab-input"
                  type="number" min={0} max={120}
                  value={age} onChange={e => setAge(e.target.value)} required
                />
              </label>

              <label className="intake-field">
                <span>Time since any knee surgery (if applicable)</span>
                <select className="rehab-input" value={surgeryRecency} onChange={e => setSurgeryRecency(e.target.value)}>
                  {SURGERY_RECENCY_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                </select>
              </label>

              <label className="intake-field">
                <span>Diagnosis / reason for rehab</span>
                <select className="rehab-input" value={diagnosis} onChange={e => setDiagnosis(e.target.value)}>
                  {DIAGNOSIS_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                </select>
              </label>

              <label className="intake-field">
                <span>Current pain level at rest (0 = none, 10 = worst pain imaginable)</span>
                <input
                  type="range" min={0} max={10} value={painLevel}
                  onChange={e => setPainLevel(e.target.value)}
                />
                <span style={{ color: 'var(--blue-light)', fontWeight: 700 }}>{painLevel} / 10</span>
              </label>

              <label className="intake-checkbox">
                <input type="checkbox" checked={swelling} onChange={e => setSwelling(e.target.checked)} />
                <span>I currently have visible swelling in the knee</span>
              </label>

              <label className="intake-checkbox">
                <input type="checkbox" checked={clearedByDoctor} onChange={e => setClearedByDoctor(e.target.checked)} required />
                <span>I have been cleared by a doctor or physical therapist to perform this exercise</span>
              </label>

              {error && (
                <div className="output-error">
                  <span style={{ color: '#e06c75' }}>⚠ {error}</span>
                </div>
              )}

              <button className="cta-btn" type="submit" disabled={submitting}>
                {submitting ? 'Checking…' : 'Continue'}
              </button>
            </form>
          )}

          {result && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <p className="block-title">What We Found</p>

              {result.warnings.length === 0 && (
                <InsightLine color="#56c596" message="No risk factors flagged from your answers." />
              )}
              {result.warnings.map((w, i) => (
                <InsightLine key={i} color={WARNING_COLOR[w.level]} message={w.message} source={w.source} />
              ))}

              {result.blocking ? (
                <div className="output-error">
                  <span style={{ color: '#e06c75' }}>
                    ⚠ Based on your answers, you should not start this session right now. Address the flagged item(s) above first.
                  </span>
                </div>
              ) : (
                <button className="cta-btn" onClick={handleContinue}>Continue to Rehab Mode</button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function InsightLine({ color, message, source }) {
  return (
    <div className="out-item out-item-neutral" style={{ borderLeftColor: color }}>
      <div className="out-item-dot" style={{ background: color }} />
      <div>
        <p className="out-item-detail" style={{ color: 'var(--text-primary)' }}>{message}</p>
        {source && (
          <a href={source} target="_blank" rel="noreferrer" style={{ fontSize: '0.75rem', color: 'var(--blue-light)' }}>
            Source
          </a>
        )}
      </div>
    </div>
  )
}
