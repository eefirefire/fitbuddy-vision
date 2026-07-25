import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { login, registerAccount } from '../authService'

export default function Auth() {
  const navigate = useNavigate()
  const location = useLocation()
  const [mode, setMode] = useState('login') // 'login' | 'register'
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const redirectTo = location.state?.redirectTo ?? '/intake'

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      if (mode === 'login') {
        await login(username, password)
      } else {
        await registerAccount(username, password)
      }
      navigate(redirectTo, { state: location.state })
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="grid-bg" style={{ alignItems: 'center', justifyContent: 'center' }}>
      <div className="hero-card" style={{ gridTemplateColumns: '1fr', maxWidth: 420, margin: '0 auto' }}>
        <div className="hero-left">
          <div className="mode-tabs">
            <button className={`mode-tab ${mode === 'login' ? 'mode-tab-active' : ''}`} onClick={() => setMode('login')} type="button">
              Log In
            </button>
            <button className={`mode-tab ${mode === 'register' ? 'mode-tab-active' : ''}`} onClick={() => setMode('register')} type="button">
              Sign Up
            </button>
          </div>

          <h1 className="hero-title" style={{ fontSize: '1.8rem' }}>
            {mode === 'login' ? 'Welcome Back.' : 'Create an Account.'}
          </h1>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <input
              className="rehab-input"
              type="text"
              placeholder="Username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              autoComplete="username"
              required
            />
            <input
              className="rehab-input"
              type="password"
              placeholder="Password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              minLength={6}
              required
            />

            {error && (
              <div className="output-error">
                <span style={{ color: '#e06c75' }}>⚠ {error}</span>
              </div>
            )}

            <button className="cta-btn" type="submit" disabled={submitting}>
              {submitting ? 'Please wait…' : mode === 'login' ? 'Log In' : 'Sign Up'}
            </button>
            <span className="cta-sub">
              Account data is stored locally for this demo — use a password you don't reuse elsewhere.
            </span>
          </form>
        </div>
      </div>
    </div>
  )
}
