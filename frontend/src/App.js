import React, { useState, useEffect } from 'react';
import { Activity, Upload, User, Menu, X, CheckCircle } from 'lucide-react';
import AiScanner from './AiScanner';
import './App.css';

export default function App() {
  const [view, setView] = useState('dashboard');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  
  const [history, setHistory] = useState(() => JSON.parse(localStorage.getItem('fitbuddy_history')) || []);
  const [isScanning, setIsScanning] = useState(false);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [file, setFile] = useState(null);

  // --- VIRTUAL COMPANION STATE ---
  const [petType, setPetType] = useState(() => localStorage.getItem('fitbuddy_pet') || null);
  const [liveSessionReps, setLiveSessionReps] = useState(0);

  // Gamification Math (History + Live Reps)
  const totalSquats = history.reduce((sum, session) => sum + (session.reps || 0), 0) + liveSessionReps;
  const petExp = totalSquats * 10;
  const petLevel = Math.floor(petExp / 500) + 1;
  const expToNextLevel = 500 - (petExp % 500);
  const expPercentage = ((petExp % 500) / 500) * 100;

  const getPetMood = () => {
    if (history.length === 0) return 'neutral';
    const lastWorkoutDate = new Date(history[0].date);
    const today = new Date();
    const diffDays = Math.ceil(Math.abs(today - lastWorkoutDate) / (1000 * 60 * 60 * 24));
    
    if (diffDays <= 1) return 'happy';
    if (diffDays === 2) return 'neutral';
    return 'sad';
  };

  const selectPet = (type) => {
    setPetType(type);
    localStorage.setItem('fitbuddy_pet', type);
  };

  const navTo = (newView) => {
    setView(newView);
    setIsMobileMenuOpen(false);
  };

  const handleFinishScan = (sessionData) => {
    const newSession = {
      ...sessionData,
      date: new Date().toISOString(),
      id: Date.now()
    };
    const updatedHistory = [newSession, ...history];
    setHistory(updatedHistory);
    localStorage.setItem('fitbuddy_history', JSON.stringify(updatedHistory));
    
    // Reset scanner states
    setLiveSessionReps(0);
    setIsScanning(false);
    setCurrentVideo(null);
    setFile(null);
  };

  return (
    <div className="app-container">
      {/* NAVBAR */}
      <nav className="navbar">
        <div className="logo" onClick={() => navTo('dashboard')}>
          <Activity size={28} className="icon-logo"/>
          <span>FitBuddy AI</span>
        </div>
        
        <div className="nav-links desktop-only">
          <button className={`nav-item ${view==='dashboard'?'active':''}`} onClick={()=>navTo('dashboard')}>Dashboard</button>
          <button className={`nav-item ${view==='calendar'?'active':''}`} onClick={()=>navTo('calendar')}>Calendar</button>
          <button className={`nav-item ${view==='history'?'active':''}`} onClick={()=>navTo('history')}>History</button>
          <button onClick={()=>navTo('profile')} className="btn-profile"><User size={16}/> Profile</button>
        </div>

        <div className="mobile-menu-btn" onClick={()=>setIsMobileMenuOpen(!isMobileMenuOpen)}>
          {isMobileMenuOpen ? <X size={28}/> : <Menu size={28}/>}
        </div>
        
        {/* MOBILE DROPDOWN */}
        {isMobileMenuOpen && (
          <div className="mobile-menu-dropdown">
            <button className={`nav-item ${view==='dashboard'?'active':''}`} onClick={()=>navTo('dashboard')}>Dashboard</button>
            <button className={`nav-item ${view==='calendar'?'active':''}`} onClick={()=>navTo('calendar')}>Calendar</button>
            <button className={`nav-item ${view==='history'?'active':''}`} onClick={()=>navTo('history')}>History</button>
            <button onClick={()=>navTo('profile')} className="btn-profile"><User size={16}/> Profile</button>
          </div>
        )}
      </nav>

      {/* DASHBOARD VIEW */}
      {view === 'dashboard' && (
        <>
          {/* --- VIRTUAL COMPANION MODULE --- */}
          <div className="pet-container">
            {!petType ? (
              <div className="pet-selector">
                <h3>Choose Your Companion</h3>
                <p>They will grow stronger as you squat.</p>
                <div className="pet-choices">
                  <button onClick={() => selectPet('cat')} className="btn-pet">🐱 Cat</button>
                  <button onClick={() => selectPet('dog')} className="btn-pet">🐶 Dog</button>
                </div>
              </div>
            ) : (
              <div className="pet-dashboard">
                <div className="pet-avatar-stage" style={{ overflow: 'hidden', borderRadius: '16px', border: '2px solid rgba(255,255,255,0.1)' }}>
                  {petType === 'cat' ? (
                    <video 
                      src="/Cat_Squat.mp4" 
                      autoPlay 
                      loop 
                      muted 
                      playsInline 
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  ) : (
                    <div className={`pet-sprite ${getPetMood()}`}>
                      {getPetMood() === 'sad' ? '🥺' : getPetMood() === 'happy' ? '🐶' : '🐕'}
                    </div>
                  )}
                </div>
                <div className="pet-stats">
                  <div className="pet-header-row">
                    <h4>Lv. {petLevel} {petType === 'cat' ? 'Iron Kitty' : 'Squat Doggo'}</h4>
                    <span className={`mood-badge ${getPetMood()}`}>
                      {getPetMood() === 'happy' ? 'Energized' : getPetMood() === 'sad' ? 'Hungry!' : 'Chilling'}
                    </span>
                  </div>
                  <div className="exp-bar-container">
                    <div className="exp-bar-fill" style={{width: `${expPercentage}%`}}></div>
                  </div>
                  <p className="exp-text">{expToNextLevel} EXP to Level {petLevel + 1}</p>
                </div>
              </div>
            )}
          </div>

          {/* HERO & UPLOAD */}
          {!isScanning && (
            <div className="hero-section">
              <h1>AI Form Coach</h1>
              <p>Real-time edge biomechanics tracking.</p>
              
              <div className="upload-box">
                <label className="file-label">
                  {/* FIX: Hidden input so it actually opens the file gallery */}
                  <input 
                    type="file" 
                    accept="video/*" 
                    style={{ display: 'none' }} 
                    onChange={(e) => {
                      const selectedFile = e.target.files[0];
                      if (selectedFile) {
                        const fileUrl = URL.createObjectURL(selectedFile);
                        setFile(fileUrl);
                        setCurrentVideo(fileUrl);
                        setIsScanning(true);
                      }
                    }} 
                  />
                  <Upload size={48} color="#00e676"/>
                  <h3>Upload a Video</h3>
                  <p>Click to browse your gallery</p>
                </label>
              </div>

              <div style={{ marginTop: '20px' }}>
                <p style={{ color: '#666' }}>— or —</p>
                <button 
                  className="btn-analyze" 
                  onClick={() => setIsScanning(true)}
                  style={{ marginTop: '10px' }}
                >
                  Start Live Camera
                </button>
              </div>
            </div>
          )}

          {/* AI SCANNER ENGINE */}
          {isScanning && (
            <div className="results-section">
              <AiScanner 
                petType={petType}
                petLevel={petLevel}
                expPercentage={expPercentage}
                onRep={(currentReps) => setLiveSessionReps(currentReps)} 
                onFinish={handleFinishScan} 
              />
            </div>
          )}
        </>
      )}

      {/* HISTORY & CALENDAR PLACEHOLDERS */}
      {view === 'history' && (
        <div className="history-section">
          <h2>Workout History</h2>
          <div className="history-grid">
            {history.length === 0 ? <p>No sessions yet.</p> : history.map(session => (
              <div key={session.id} className="history-card">
                <div className="history-info">
                  <h3>{new Date(session.date).toLocaleDateString()}</h3>
                  <p className="rep-badge">{session.reps} Reps</p>
                  <p style={{color: '#aaa', fontSize: '0.9rem'}}>{session.feedback}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {view === 'calendar' && (
        <div className="calendar-section">
          <h2>Training Calendar</h2>
          <p style={{color: '#666'}}>Calendar interface is currently under construction.</p>
        </div>
      )}

      {view === 'profile' && (
        <div className="profile-section">
          <h2>Profile Settings</h2>
          <button 
            className="btn-reset" 
            onClick={() => {
              localStorage.removeItem('fitbuddy_pet');
              setPetType(null);
              navTo('dashboard');
            }}
          >
            Release Companion & Choose Another
          </button>
        </div>
      )}
    </div>
  );
}