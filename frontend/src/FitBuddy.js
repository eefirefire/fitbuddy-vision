import React, { useState } from 'react';

const FitBuddy = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [exerciseMode, setExerciseMode] = useState("squat");
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setProcessedVideoUrl(null); 
  };

  const handleModeChange = (mode) => {
    setExerciseMode(mode);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('mode', exerciseMode); 

    try {
      const response = await fetch('http://localhost:8000/upload_video', {
        method: 'POST',
        body: formData,
      });

      const videoBlob = await response.blob();
      const videoUrl = URL.createObjectURL(videoBlob);
      setProcessedVideoUrl(videoUrl);
    } catch (error) {
      console.error("Error processing video:", error);
      alert("Something went wrong with the AI processing.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.appContainer}>
      
      {/* LEFT COLUMN: Controls & Settings */}
      <div style={styles.sidebar}>
        <div style={styles.brand}>
          <h1 style={styles.logo}>🤖 FitBuddy</h1>
          <p style={styles.tagline}>AI Personal Trainer</p>
        </div>

        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>1. Choose Exercise</h3>
          <div style={styles.modeContainer}>
            <button 
              style={exerciseMode === 'squat' ? styles.modeButtonActive : styles.modeButton}
              onClick={() => handleModeChange('squat')}
            >
              <span style={styles.emoji}>🏋️‍♂️</span> Squats
            </button>
            <button 
              style={exerciseMode === 'pushup' ? styles.modeButtonActive : styles.modeButton}
              onClick={() => handleModeChange('pushup')}
            >
              <span style={styles.emoji}>💪</span> Push-ups
            </button>
          </div>
        </div>

        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>2. Upload Video</h3>
          <div style={styles.uploadBox}>
            <input 
              type="file" 
              accept="video/*" 
              onChange={handleFileChange} 
              style={styles.fileInput}
              id="file-upload"
            />
            <label htmlFor="file-upload" style={styles.uploadLabel}>
              {selectedFile ? selectedFile.name : "📁 Browse Files"}
            </label>
          </div>
        </div>

        <button 
          onClick={handleUpload} 
          disabled={!selectedFile || isLoading}
          style={isLoading ? styles.analyzeButtonLoading : styles.analyzeButton}
        >
          {isLoading ? "🧠 PROCESSING..." : "🚀 ANALYZE FORM"}
        </button>
      </div>

      {/* RIGHT COLUMN: Video Player & Output */}
      <div style={styles.mainContent}>
        <div style={styles.videoCard}>
          <div style={styles.videoHeader}>
            <span style={styles.statusIndicator}></span> 
            {isLoading ? "AI Vision: Analyzing Frame Data..." : "AI Vision: Live Output"}
          </div>
          
          <div style={styles.playerWrapper}>
            {processedVideoUrl ? (
              <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                <video controls autoPlay style={styles.videoPlayer}>
                  <source src={processedVideoUrl} type="video/mp4" />
                </video>
                <a href={processedVideoUrl} download={`FitBuddy_${exerciseMode}_Results.mp4`} style={styles.downloadBtn}>
                  ⬇️ Download Analysis
                </a>
              </div>
            ) : (
              <div style={styles.placeholder}>
                {isLoading ? (
                  <div style={styles.spinnerContainer}>
                    <div className="spinner" style={styles.spinner}></div>
                    <p>Crunching the numbers...</p>
                  </div>
                ) : (
                  <div>
                    <h2 style={styles.placeholderTitle}>Awaiting Video Input</h2>
                    <p style={styles.placeholderText}>Select an exercise mode and upload your workout to begin tracking.</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

    </div>
  );
};

// --- MODERN DASHBOARD CSS STYLES ---
const styles = {
  appContainer: {
    display: 'flex',
    height: '100vh',
    backgroundColor: '#121212', // Dark background
    color: '#ffffff',
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    overflow: 'hidden',
  },
  sidebar: {
    width: '320px',
    backgroundColor: '#1e1e1e',
    padding: '30px',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '4px 0 15px rgba(0,0,0,0.5)',
    zIndex: 2,
  },
  brand: {
    marginBottom: '40px',
  },
  logo: {
    fontSize: '32px',
    fontWeight: '800',
    margin: 0,
    background: 'linear-gradient(90deg, #00C853, #64DD17)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  tagline: {
    margin: '5px 0 0 0',
    color: '#888',
    fontSize: '14px',
    textTransform: 'uppercase',
    letterSpacing: '1px',
  },
  section: {
    marginBottom: '30px',
  },
  sectionTitle: {
    fontSize: '12px',
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '1px',
    marginBottom: '15px',
  },
  modeContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  modeButton: {
    padding: '15px',
    fontSize: '16px',
    fontWeight: '600',
    backgroundColor: '#2a2a2a',
    color: '#fff',
    border: '2px solid transparent',
    borderRadius: '10px',
    cursor: 'pointer',
    textAlign: 'left',
    display: 'flex',
    alignItems: 'center',
    transition: 'all 0.2s',
  },
  modeButtonActive: {
    padding: '15px',
    fontSize: '16px',
    fontWeight: '600',
    backgroundColor: '#1e382b',
    color: '#00e676',
    border: '2px solid #00C853',
    borderRadius: '10px',
    cursor: 'pointer',
    textAlign: 'left',
    display: 'flex',
    alignItems: 'center',
  },
  emoji: {
    fontSize: '24px',
    marginRight: '15px',
  },
  uploadBox: {
    position: 'relative',
  },
  fileInput: {
    display: 'none', // Hiding the default ugly file input
  },
  uploadLabel: {
    display: 'block',
    padding: '15px',
    backgroundColor: '#2a2a2a',
    color: '#fff',
    borderRadius: '10px',
    textAlign: 'center',
    cursor: 'pointer',
    border: '2px dashed #444',
    fontWeight: '500',
    transition: 'all 0.2s',
  },
  analyzeButton: {
    marginTop: 'auto',
    padding: '18px',
    fontSize: '16px',
    fontWeight: '800',
    backgroundColor: '#00C853',
    color: '#fff',
    border: 'none',
    borderRadius: '10px',
    cursor: 'pointer',
    boxShadow: '0 4px 15px rgba(0, 200, 83, 0.4)',
    textTransform: 'uppercase',
    letterSpacing: '1px',
  },
  analyzeButtonLoading: {
    marginTop: 'auto',
    padding: '18px',
    fontSize: '16px',
    fontWeight: '800',
    backgroundColor: '#444',
    color: '#888',
    border: 'none',
    borderRadius: '10px',
    cursor: 'not-allowed',
    textTransform: 'uppercase',
    letterSpacing: '1px',
  },
  mainContent: {
    flex: 1,
    padding: '40px',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#121212',
  },
  videoCard: {
    width: '100%',
    maxWidth: '900px',
    backgroundColor: '#1e1e1e',
    borderRadius: '16px',
    overflow: 'hidden',
    boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
  },
  videoHeader: {
    padding: '15px 20px',
    backgroundColor: '#252525',
    color: '#aaa',
    fontSize: '14px',
    fontWeight: '600',
    display: 'flex',
    alignItems: 'center',
    borderBottom: '1px solid #333',
  },
  statusIndicator: {
    width: '8px',
    height: '8px',
    backgroundColor: '#00C853',
    borderRadius: '50%',
    marginRight: '10px',
    boxShadow: '0 0 8px #00C853',
  },
  playerWrapper: {
    width: '100%',
    aspectRatio: '16 / 9',
    backgroundColor: '#000',
    position: 'relative',
  },
  videoPlayer: {
    width: '100%',
    height: '100%',
    objectFit: 'contain',
  },
  placeholder: {
    width: '100%',
    height: '100%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    flexDirection: 'column',
    color: '#555',
  },
  placeholderTitle: {
    fontSize: '24px',
    color: '#fff',
    marginBottom: '10px',
  },
  placeholderText: {
    color: '#888',
  },
  downloadBtn: {
    position: 'absolute',
    top: '20px',   // CHANGED: Was 'bottom: 20px'
    right: '20px', // Kept 'right'
    padding: '10px 20px',
    backgroundColor: 'rgba(0, 200, 83, 0.9)',
    color: '#fff',
    textDecoration: 'none',
    borderRadius: '8px',
    fontWeight: '600',
    backdropFilter: 'blur(5px)',
    zIndex: 10, // Ensures it sits on top of the video
    boxShadow: '0 4px 6px rgba(0,0,0,0.3)', // Adds a nice shadow for visibility
  },
  spinnerContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    color: '#00C853',
  },
  spinner: {
    border: '4px solid rgba(255, 255, 255, 0.1)',
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    borderLeftColor: '#00C853',
    animation: 'spin 1s linear infinite',
    marginBottom: '15px',
  }
};

export default FitBuddy;