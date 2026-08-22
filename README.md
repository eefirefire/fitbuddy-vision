# FitBuddy

**Hybrid edge-cloud computer vision for exercise coaching and orthopedic rehabilitation — no wearables, no lab, just a laptop webcam.**

Built by [Anavin Srison (Evin)](https://github.com/eefirefire) — engineering & implementation, and Eva — research design & clinical grounding · TISIIF 2026 & TICTA 2026 Student Project

---

## What it does

FitBuddy runs two parallel pipelines, each assigned to the compute tier appropriate for its task:

| Mode | Input | Where it runs | What you get |
|---|---|---|---|
| **Exercise Mode** | Video upload | Cloud (Gemini Vision) | Structured squat coaching — rep count, faults, coaching notes |
| **Rehab Mode** | Webcam or video upload | On-device (MediaPipe) | Real-time joint angle, rep count, extension deficit, coaching cues |

No video is stored on any server. Exercise mode sends 15 keyframes extracted client-side in the browser; rehab mode never leaves the device.

One-page overview: [`poster/fitbuddy_poster.pdf`](poster/fitbuddy_poster.pdf)

---

## Validated accuracy

Tested against OnlineProtractor.com (digital goniometer) across 17 knee-extension repetitions:

| Condition | n | MAE | 95% CI | R² |
|---|---|---|---|---|
| Perfect alignment | 6 | **2.93°** | [0.95, 4.91] | — |
| Faulty rep | 5 | **3.36°** | [0.00, 8.30]† | — |
| Off-angle camera | 6 | **6.64°** | [1.84, 11.45] | — |
| **Combined perpendicular** | **11** | **3.12°** | [1.21, 5.04] | **0.86** |

† CI lower bound truncated at 0; MAE cannot be negative. Wide interval reflects n=5 with one high-error rep.

Off-angle placement roughly doubles error due to 3D-to-2D perspective foreshortening. The built-in `CameraAlignmentChecker` detects this in real time and prompts the user to correct their setup.

Raw data and chart generation: `eda/Validation/`

---

## Performance (Rehab Mode)

Benchmarked on a real clip via `Scripts/benchmark_rehab_pipeline.py`:

- **~15–17 FPS** throughput
- **~65 ms** average per-frame latency

---

## Architecture

```
REHAB MODE
  Webcam or video upload → MediaPipe BlazePose (on-device) → joint angles → rep counter
  → extension deficit · coaching cues · descent speed · trunk compliance
  No cloud dependency. Live mode runs at 15–17 FPS.

EXERCISE MODE
  Video upload → 15 keyframes extracted in-browser (Canvas API, <50MB RAM)
              → Gemini Vision API (CoT + Self-Reflection prompt)
              → structured JSON coaching feedback
  No video stored on any server.
```

**Prompt benchmarking (Exercise Mode):** Five prompting strategies were evaluated across 40 annotated ground-truth profiles, scored by an independent LLM judge (Gemini 2.5 Flash):

| Rank | Strategy | Avg. score (0–10) |
|---|---|---|
| 1 | **Self-Reflection + CoT** (production) | **7.40** |
| 2 | Balanced Few-Shot | 6.75 |
| 3 | Three-Shot | 5.81 |
| 4 | CoT-Verification (no reflection) | 5.70 |
| 5 | Zero-Shot baseline | 4.46 |

+66% alignment gain over zero-shot baseline through prompt design alone, no fine-tuning.

---

## Project layout

```
Scripts/
  rehab_knee_extension.py   — Flask server, MediaPipe pipeline, rehab session logic
  rehab_auth.py             — account system (salted hashes, session cookies)
  server.py                 — FastAPI server, Gemini exercise analysis
  test_rehab_knee_extension.py — 20 unit tests for the rehab pipeline
  benchmark_rehab_pipeline.py  — FPS / latency benchmark
  requirements.txt

squat-frontend/
  src/pages/
    Landing.jsx             — mode selector
    Dashboard.jsx           — exercise mode upload + Gemini feedback
    RehabPanel.jsx          — live rehab tracking UI
    Auth.jsx                — login / register
    Intake.jsx              — pre-exercise safety intake

eda/
  Validation/               — raw validation data, chart generation scripts

video tests/
  README.md                 — real (not synthetic) footage used to regression-
                               test the rehab pipeline end-to-end, with results
```

---

## Setup

**Backend:**

```bash
cd Scripts
pip install -r requirements.txt
python rehab_knee_extension.py   # rehab server → localhost:5050
python server.py                 # exercise server (optional)
```

**Frontend:**

```bash
cd squat-frontend
npm install
cp .env.example .env             # add your VITE_GEMINI_API_KEY
npm run dev                      # → localhost:5173
```

---

## Safety design (Rehab Mode)

Rehab Mode requires account login and a pre-exercise intake before every session. Clinical thresholds are sourced from published guidance:

| Signal | Threshold | Source |
|---|---|---|
| Pain level | 0–3 proceed · 4–6 caution · 7+ blocked | PT pain traffic-light rule |
| Medical clearance | Required to proceed | Intake requirement |
| Age | 65+ → fall-risk caution | CDC STEADI |
| Visible swelling | Caution — scale back ROM | Standard PT practice |
| Recent surgery | 0–4 weeks post-op + active diagnosis → caution | Defers to surgeon/PT |

FitBuddy is a screening aid, not a clinical tool. It does not replace a doctor or physical therapist.

---

## License

MIT — see [LICENSE](LICENSE).
