# FitBuddy AI - Project Context

FitBuddy AI is a gamified fitness application that leverages real-time AI biomechanics tracking to provide form coaching for exercises (currently focused on squats). The application features a virtual companion that "levels up" as the user completes repetitions, adding a layer of gamification to the fitness experience.

## Project Overview

- **Core Purpose:** Real-time exercise form coaching and rep counting using edge AI.
- **Main Technologies:**
    - **Framework:** React 19
    - **AI/ML:** TensorFlow.js with the **MoveNet** pose detection model.
    - **Mobile:** Capacitor (Android & iOS) for native mobile deployment.
    - **Styling:** Vanilla CSS with a "Glassmorphism" and "Neon" aesthetic (Dark mode).
    - **Icons:** Lucide-react.
    - **Voice:** Capacitor Text-to-Speech with a browser fallback for rep counting and feedback.
- **Architecture:**
    - `src/App.js`: The central hub for state management (companion levels, workout history, view routing).
    - `src/AiScanner.js`: The "AI Engine" component. It handles the webcam feed, initializes the TensorFlow model, and executes the squat analysis logic (angle calculations and state tracking).
    - `src/FitBuddy.js`: A secondary/alternative UI that supports video uploads to a backend for analysis (likely a legacy or experimental feature compared to the live `AiScanner`).
    - `public/`: Contains assets like `Cat_Squat.mp4` used for the virtual companion.

## Building and Running

### Development
- **Start Web App:** `npm start` (Runs at `http://localhost:3000`).
- **Run Tests:** `npm test`.

### Production
- **Build Web App:** `npm run build`.
- **Capacitor Sync:** `npx cap sync` (To update native Android/iOS projects with the latest build).

### Mobile Specifics
- **Android Project:** Located in `/android`.
- **iOS Project:** Located in `/ios`.
- **Config:** `capacitor.config.ts`.

## Development Conventions

- **Theming:** Strictly follows the CSS variables defined in `src/App.css` (e.g., `--primary: #00e676`, `--bg-dark: #050505`).
- **State Management:** Uses React `useState` and `useEffect`. Persistent data (history, companion state) is stored in `localStorage`.
- **AI Logic:**
    - Uses **MoveNet (SinglePose Lightning)** for performance on mobile/edge devices.
    - Squat detection is based on the angle calculated between the hip, knee, and ankle keypoints.
    - Reps are counted when the user transitions from a "down" state (angle < 90°) back to an "up" state (angle > 160°).
- **Responsive Design:** Utilizes a mobile-first approach with a dedicated mobile menu dropdown for smaller screens.

## Key Files

- `src/App.js`: Main application logic and routing.
- `src/AiScanner.js`: Real-time AI pose detection and feedback engine.
- `src/App.css`: Global styles and design tokens.
- `package.json`: Dependency management and scripts.
- `capacitor.config.ts`: Capacitor configuration for mobile builds.
