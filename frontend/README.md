# FitBuddy AI - Real-Time Fitness Coaching & Gamification

FitBuddy AI is a cutting-edge, gamified fitness application designed to provide real-time biomechanics tracking and form coaching. By leveraging edge AI and computer vision, FitBuddy helps users perfect their exercise form while adding an engaging layer of gamification through a virtual fitness companion.

---

## 🚀 Key Features

- **Real-Time AI Coaching:** Instant feedback on squat form using your device's camera.
- **Rep Counting:** Automated and accurate rep tracking with voice feedback.
- **Virtual Companion:** A "level-up" system where your virtual pet (Cat or Dog) gains EXP with every rep.
- **Video Analysis Mode:** Upload recorded workouts to a backend for deep analysis and detailed form reports.
- **Mobile First:** Built with Capacitor for a seamless experience on Android and iOS.
- **Voice Feedback:** Hybrid engine (Native + Web) that speaks rep counts and coaching cues.
- **Glassmorphic UI:** Modern, dark-themed interface with neon accents for a high-tech feel.

---

## 🧠 How It Works: Under the Hood

FitBuddy AI uses a dual-engine architecture to balance real-time performance with deep biomechanical analysis.

### 1. Real-Time Analysis (The "How" of Live Coaching)
The frontend engine (in `AiScanner.js`) is designed for zero-latency feedback.

*   **The Vision Loop:** Uses `requestAnimationFrame` to create a continuous detection loop. Each frame from the `react-webcam` is passed to **MoveNet (SinglePose Lightning)**, which returns 17 keypoints (coordinates + confidence scores).
*   **Geometric Logic:** To analyze a squat, we calculate the interior angle of the knee.
    *   **Formula:** We use `Math.atan2` to find the slope between the Hip-Knee and Knee-Ankle, then convert the difference from radians to degrees.
    *   **Vector Math:** `angle = |atan2(Ankle.y - Knee.y, Ankle.x - Knee.x) - atan2(Hip.y - Knee.y, Hip.x - Knee.x)|`
*   **The State Machine (Rep Counting):** To prevent "double-counting" or jitter, we use a debounced state machine:
    1.  **State `UP`:** Default state.
    2.  **State `DOWN`:** Triggered when the knee angle drops below **90°**.
    3.  **Completion:** A rep is only counted when the user returns to the `UP` state (angle > **160°**).
*   **Voice Synthesis:** Uses the `@capacitor-community/text-to-speech` plugin for native mobile voices, with a `window.speechSynthesis` fallback for web browsers.

### 2. Deep Analysis (The "How" of Backend Processing)
The backend (in `api.py`) handles complex ML classification that is too heavy for mobile browsers.

*   **Precision Tracking:** Uses **YOLOv8-pose** (Ultralytics) to extract keypoints. YOLO is significantly more robust than MoveNet for varying distances and backgrounds.
*   **The "Angle Brain" (ML Classifier):** 
    *   Instead of just checking one angle, the backend extracts a **feature vector** for every frame: `[Left/Right Knee, Left/Right Hip, Torso Lean, Spine Angle]`.
    *   These features are fed into a **Random Forest Classifier** (`fitbuddy_angle_brain.pkl`) which was trained on thousands of labeled squat samples.
    *   The model "votes" on the form for every frame; the most frequent vote during the "down" phase becomes the final form label (e.g., "Knee Cave" or "Heel Lift").
*   **Quality Gate Algorithm:** Before analysis, the video passes through a "Gatekeeper" that checks:
    *   **Blur Detection:** Uses a Laplacian variance threshold (Sharpness < 35 = Reject).
    *   **Visibility:** Ensures leg keypoints have high confidence scores (>0.5).
    *   **Stability:** Measures the distance keypoints move between frames to detect camera shake.

### 3. Gamification & Data Flow
*   **State Management:** The global application state (Levels, EXP, History) is managed in `App.js`.
*   **Persistence:** Every successful rep sends a signal from `AiScanner` to `App.js`, which updates `localStorage`. This ensures your pet's progress is saved even if you close the app.
*   **Dynamic UI:** The "Glassmorphism" effect is achieved via CSS `backdrop-filter: blur()` and semi-transparent RGBA colors, providing a premium feel without heavy assets.

---

## 🛠 Tech Stack

### Frontend
- **Framework:** React 19
- **AI/ML:** TensorFlow.js, MoveNet
- **Mobile:** Capacitor (Android & iOS)
- **Icons:** Lucide-react
- **Styling:** Vanilla CSS (Glassmorphism / Neon aesthetic)

### Backend (Analysis Engine)
- **Framework:** FastAPI (Python)
- **AI/ML:** Ultralytics YOLOv8, Scikit-Learn (Random Forest)
- **Video Processing:** OpenCV
- **Server:** Uvicorn

---

## 📂 Project Structure

```text
├── frontend/
│   ├── src/
│   │   ├── AiScanner.js      # Live detection & Geometry logic
│   │   ├── App.js            # Main Hub & Gamification logic
│   │   ├── FitBuddy.js       # Backend Integration UI
│   │   └── App.css           # Styling & Design Tokens
│   ├── android/              # Native Android wrapper
│   └── ios/                  # Native iOS wrapper
└── backend/
    ├── api.py                # FastAPI & YOLOv8 Processing
    ├── train_angles.py       # ML Training Pipeline
    └── math_utils.py         # Backend Angle Logic
```

---

## 🚦 Getting Started

### Prerequisites
- Node.js (v18+)
- Python 3.9+
- Android Studio / Xcode

### 1. Frontend Setup
```bash
cd frontend
npm install
npm start
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python api.py
```

---

## 📜 License
MIT License - see the LICENSE file for details.
