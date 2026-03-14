import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { Activity, CheckCircle, Volume2 } from 'lucide-react';
import { TextToSpeech } from '@capacitor-community/text-to-speech';

export default function AiScanner({ onFinish, onRep, petType, petLevel, expPercentage }) {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [isReady, setIsReady] = useState(false);
  const [reps, setReps] = useState(0);
  const [feedback, setFeedback] = useState("Stand in frame");
  
  const squatState = useRef('up'); 
  const isAlive = useRef(true); 

  // --- HYBRID VOICE ENGINE (Native + Web Fallback) ---
  const speakText = async (text) => {
    const message = text.toString();
    try {
      // Try Native Android Voice First
      await TextToSpeech.speak({
        text: message,
        lang: 'en-US',
        rate: 1.0,
        volume: 1.0,
        category: 'ambient'
      });
    } catch (error) {
      // Fallback to Browser Voice for PC/npm start
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(message);
        window.speechSynthesis.speak(utterance);
      }
    }
  };

  useEffect(() => {
    let detector;
    let animId;
    isAlive.current = true;

    const runAI = async () => {
      await tf.ready();
      detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );
      if (isAlive.current) {
        setIsReady(true);
        detectPose();
      }
    };

    const detectPose = async () => {
      if (!isAlive.current || !webcamRef.current || !canvasRef.current) return; 

      if (webcamRef.current.video && webcamRef.current.video.readyState === 4) {
        const video = webcamRef.current.video;
        const { videoWidth, videoHeight } = video;

        webcamRef.current.video.width = videoWidth;
        webcamRef.current.video.height = videoHeight;
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        try {
          const poses = await detector.estimatePoses(video);
          if (canvasRef.current && isAlive.current) {
            drawCanvas(poses, videoWidth, videoHeight);
            analyzeSquat(poses);
          }
        } catch (error) {
          console.log("AI reading error:", error);
        }
      }
      if (isAlive.current) animId = requestAnimationFrame(detectPose);
    };

    runAI();

    return () => {
      isAlive.current = false;
      cancelAnimationFrame(animId);
      if (detector) detector.dispose();
      TextToSpeech.stop().catch(() => {});
    };
  }, []);

  const calculateAngle = (a, b, c) => {
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs((radians * 180.0) / Math.PI);
    return angle > 180.0 ? 360 - angle : angle;
  };

  const analyzeSquat = (poses) => {
    if (!poses || poses.length === 0) return setFeedback("No person detected");

    const keypoints = poses[0].keypoints;
    const hip = keypoints.find(k => k.name === 'left_hip');
    const knee = keypoints.find(k => k.name === 'left_knee');
    const ankle = keypoints.find(k => k.name === 'left_ankle');

    if (hip && knee && ankle && hip.score > 0.5 && knee.score > 0.5 && ankle.score > 0.5) {
      const angle = calculateAngle(hip, knee, ankle);

      if (angle > 160) {
        if (squatState.current === 'down') {
          setReps(prev => {
            const newReps = prev + 1;
            if (onRep) onRep(newReps); 
            speakText(newReps); // Count rep out loud
            return newReps;
          });
          setFeedback("Good depth. Go again!");
        }
        squatState.current = 'up';
      }

      if (angle < 90) {
        if (squatState.current !== 'down') {
          speakText("Push up!"); // Encouragement
          setFeedback("Hold... and UP!");
        }
        squatState.current = 'down';
      } else if (angle < 140 && squatState.current === 'up') {
        setFeedback("Lower...");
      }
    } else {
      setFeedback("Move back so full legs are visible.");
    }
  };

  const drawCanvas = (poses, videoWidth, videoHeight) => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (poses.length > 0) {
      const keypoints = poses[0].keypoints;
      keypoints.forEach((point) => {
        if (point.score > 0.5) {
          ctx.beginPath();
          ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "#00e676";
          ctx.fill();
        }
      });
    }
  };

  return (
    <div className="ai-scanner-container">
      <div className="scanner-header">
        {/* MINI PET TRACKER */}
        {petType && (
          <div className="mini-pet-tracker" style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(255,255,255,0.1)', padding: '5px 15px', borderRadius: '12px' }}>
            {petType === 'cat' ? (
              <video 
                key="scanner-cat-video"
                src="Cat_Squat.mp4" 
                autoPlay loop muted playsInline preload="auto"
                style={{ width: '50px', height: '50px', borderRadius: '8px', objectFit: 'cover' }}
              />
            ) : (
              <span style={{ fontSize: '2rem' }}>🐶</span>
            )}
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ color: '#fff', fontWeight: 'bold', fontSize: '0.9rem' }}>Lv. {petLevel}</span>
              <div style={{ width: '80px', height: '6px', background: '#000', borderRadius: '4px', marginTop: '4px' }}>
                <div style={{ width: `${expPercentage}%`, height: '100%', background: '#00e676', borderRadius: '4px' }}></div>
              </div>
            </div>
          </div>
        )}

        <div className="rep-counter">
          <span className="rep-number">{reps}</span><span className="rep-label">REPS</span>
        </div>
        
        <div style={{ display: 'flex', gap: '8px' }}>
          {/* TEST VOICE BUTTON - Use this to "wake up" audio on Android */}
          <button 
            className="btn-end-session" 
            style={{ background: '#2979ff', padding: '10px' }}
            onClick={() => speakText("Voice active")}
          >
            <Volume2 size={18} />
          </button>

          <button className="btn-end-session" onClick={async () => {
            await TextToSpeech.stop().catch(() => {});
            onFinish({ reps, accuracy: 95, feedback });
          }}>
            <CheckCircle size={18} /> Finish
          </button>
        </div>
      </div>

      <div className="video-container-glass" style={{ minHeight: '400px', position: 'relative' }}>
        {!isReady && (
          <div className="loading-overlay">
            <Activity size={40} className="spinner-icon" color="#00e676"/>
            <p>Initializing AI Core...</p>
          </div>
        )}
        <Webcam ref={webcamRef} style={{ position: "absolute", left: 0, right: 0, width: "100%", height: "100%", objectFit: "cover", zIndex: 9 }} />
        <canvas ref={canvasRef} style={{ position: "absolute", left: 0, right: 0, width: "100%", height: "100%", objectFit: "cover", zIndex: 10 }} />
      </div>

      <div className="live-feedback-bar">
        <h3>AI Coach:</h3>
        <p className={squatState.current === 'down' ? 'text-green' : 'text-blue'}>{feedback}</p>
      </div>
    </div>
  );
}