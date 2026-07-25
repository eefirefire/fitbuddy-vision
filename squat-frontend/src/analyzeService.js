import { GoogleGenAI } from '@google/genai'

// Read from .env (gitignored) — see .env.example for the expected variable
// name. Never hardcode a real key here; this file is committed to a public repo.
const apiKey = import.meta.env.VITE_GEMINI_API_KEY
if (!apiKey) {
  throw new Error('VITE_GEMINI_API_KEY is not set. Copy .env.example to .env and fill in your key.')
}
const client = new GoogleGenAI({ apiKey })
const ANALYST_MODEL_ID = 'models/gemini-3.1-flash-lite-preview'

// ── Exact prompts from few-shot.py ────────────────────────────────────────────

const BASE_SYSTEM_INSTRUCTION = `You are an Expert Biomechanical Analyst specializing in powerlifting squat analysis.
You will be provided with a chronological grid of visual frames spanning an entire training set.
Analyze the global movement patterns, count the exact repetition scope across the whole video, and output a precise form rating based on our official rubric.
`

const COT_INSTRUCTION_APPEND = `
### 🧠 STRUCTURAL CHAIN-OF-THOUGHT PROTOCOL:
Before building the final JSON payload, you MUST perform a critical validation review.
Evaluate each deduction item based on the full timeline sequence. If an error is a minor stabilization anomaly, do not penalize.
`

// CoT-Verification-Shot strategy: BASE + COT appended together (line 218-220 in few-shot.py)
const SYSTEM_INSTRUCTION = BASE_SYSTEM_INSTRUCTION + COT_INSTRUCTION_APPEND

// ── Hardcoded one-shot exemplar (ground-truth from results/structured_feedback) ──
// In the browser we can't load videos from disk, so we inject the verified
// JSON answer + coaching text exactly as few-shot.py does for each exemplar shot.

const EXEMPLAR_1_JSON = {
  exercise: "barbell_back_squat",
  rep_count: 5,
  estimated_load: "moderate",
  overall_rating: 7,
  good: [
    { aspect: "Depth", detail: "Consistent, controlled depth that effectively hits parallel or slightly below." },
    { aspect: "Bar Path", detail: "The bar maintains a relatively vertical path directly over the midfoot throughout the reps." }
  ],
  needs_improvement: [
    { aspect: "Lumbar & Pelvic Stability", detail: "Observable 'butt wink' (posterior pelvic tilt) at the bottom of the squat, likely due to tight hamstrings or reaching depth limit." },
    { aspect: "Upper Body", detail: "The torso angle leans forward significantly as fatigue sets in during the later reps, increasing stress on the lower back." },
    { aspect: "Heel Stability", detail: "Slight rocking toward the toes is visible in some repetitions; ensure weight distribution remains centred on the midfoot." }
  ],
  visual_anchors: { start_timestamp: "00:00", end_timestamp: "00:20" }
}

const EXEMPLAR_1_TEXT = `Low-bar back squat. Lifter achieves consistent depth with a vertical bar path.
Main issues: posterior pelvic tilt (butt wink) at the hole due to limited posterior chain mobility,
progressive forward torso lean as fatigue accumulates, and mild toe-weight shift in later reps.
Core bracing and hamstring flexibility work recommended.`

// ── Exact task instruction string from few-shot.py line 250 ──────────────────
const TASK_INSTRUCTION = "### 🎯 CURRENT RUN ASSESSMENT TARGET:\nEvaluate this comprehensive frame matrix timeline sequence blindly. Output raw JSON."

// ── Exact reflection prompt from few-shot.py lines 112-119 ──────────────────
function buildReflectionPrompt(rawDraftText) {
  return `You are an Expert Biomechanical Verification Auditor.
Review your draft squat assessment carefully to eliminate false positives or alignment anomalies.
Ensure your parsed repetition count matches the full frame matrix timeline scope.

[DRAFT ASSESSMENT]:
${rawDraftText}

Return a revised, structurally pristine JSON block following the identical schema rules. Output ONLY JSON.`
}

// ── parse_to_dict_safely (ported from few-shot.py lines 46-54) ───────────────
function parseToDictSafely(textPayload) {
  try {
    let clean = textPayload.trim()
    // Strip ```json or ``` fences
    clean = clean.replace(/^```(?:json)?/i, '').replace(/```$/, '').trim()
    return JSON.parse(clean)
  } catch {
    return { parsing_error: true, raw_text: textPayload }
  }
}

// ── extract_temporal_frame_matrices (browser port of few-shot.py lines 56-79) ─
// OpenCV is replaced with the HTML5 Video + Canvas API.
// Logic is identical: sample frames evenly across the full duration.
function extractTemporalFrameMatrices(videoFile, maxFrames = 15) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(videoFile)
    const video = document.createElement('video')
    video.src = url
    video.muted = true
    video.preload = 'metadata'
    video.crossOrigin = 'anonymous'

    const canvas = document.createElement('canvas')
    canvas.width = 256
    canvas.height = 256
    const ctx = canvas.getContext('2d')

    video.addEventListener('loadedmetadata', () => {
      const duration = video.duration
      if (!duration || duration === Infinity) {
        URL.revokeObjectURL(url)
        return reject(new Error('Could not read video duration.'))
      }

      // Mirror the Python interval logic: evenly spaced timestamps
      const timestamps = Array.from({ length: maxFrames }, (_, i) =>
        (duration / maxFrames) * i
      )

      const frames = []
      let idx = 0

      function seekNext() {
        if (idx >= timestamps.length) {
          URL.revokeObjectURL(url)
          return resolve(frames)
        }
        video.currentTime = timestamps[idx]
      }

      video.addEventListener('seeked', () => {
        ctx.drawImage(video, 0, 0, 256, 256)
        frames.push(canvas.toDataURL('image/jpeg', 0.8))
        idx++
        seekNext()
      })

      seekNext()
    })

    video.addEventListener('error', () => {
      URL.revokeObjectURL(url)
      reject(new Error('Failed to load video file.'))
    })
  })
}

// ── Helper: convert base64 dataURL → inlineData part for Gemini JS SDK ───────
function frameToInlinePart(dataUrl) {
  return {
    inlineData: {
      mimeType: 'image/jpeg',
      data: dataUrl.split(',')[1],
    },
  }
}

// ── Main export ───────────────────────────────────────────────────────────────
// Strategy: CoT-Verification-Shot (shots=1, cot=True) + reflection pass
// Contents layout mirrors few-shot.py lines 218-258 exactly:
//   [system+cot] → [exemplar frames*] → [exemplar answer text] → [target frames] → [task instruction]
// * In the browser we have no disk access for exemplar videos, so we inject the
//   verified ground-truth JSON + text directly — same information, no visual frames.
// Then a second reflection call mirrors run_reflection_pass (lines 111-124).

export async function analyzeLift(videoFile, sam3Enabled = false) {
  // Step 1: extract temporal frame matrix (mirrors few-shot.py lines 243-247)
  const targetFrames = await extractTemporalFrameMatrices(videoFile, 15)
  if (targetFrames.length === 0) {
    throw new Error('Frame vector extraction failure.')
  }

  // Step 2: build contents_payload (mirrors lines 218-250)
  const parts = [
    // System instruction with CoT appended
    { text: SYSTEM_INSTRUCTION },

    // === EXEMPLAR 1 ANSWER METRICS === (mirrors lines 234-239)
    // (No exemplar video frames in browser — inject ground-truth answer directly)
    {
      text:
        `=== EXEMPLAR 1 ANSWER METRICS ===\n` +
        `Ground-Truth JSON:\n${JSON.stringify(EXEMPLAR_1_JSON, null, 2)}\n` +
        `Text Feedback:\n${EXEMPLAR_1_TEXT}\n`,
    },

    // Target frames (mirrors contents_payload.extend(target_frames), line 249)
    ...targetFrames.map(frameToInlinePart),

    // SAM3 annotation (injected before task instruction when enabled)
    ...(sam3Enabled
      ? [{ text: '[SAM3 MODE] This video has been pre-processed with skeleton keypoint overlays. Weight joint-angle evidence more heavily than raw pixel appearance when assessing depth and alignment.' }]
      : []),

    // Exact task instruction string (line 250)
    { text: TASK_INSTRUCTION },
  ]

  // Step 3: first pass — analyst call (mirrors lines 253-255)
  const firstResp = await client.models.generateContent({
    model: ANALYST_MODEL_ID,
    contents: [{ role: 'user', parts }],
  })
  let rawAnalystText = firstResp.candidates?.[0]?.content?.parts?.[0]?.text ?? ''

  // Step 4: reflection pass — mirrors run_reflection_pass (lines 111-124)
  try {
    const reflectionResp = await client.models.generateContent({
      model: ANALYST_MODEL_ID,
      contents: [{ role: 'user', parts: [{ text: buildReflectionPrompt(rawAnalystText) }] }],
    })
    const reflectedText = reflectionResp.candidates?.[0]?.content?.parts?.[0]?.text ?? ''
    if (reflectedText) rawAnalystText = reflectedText
  } catch {
    // Reflection failure is non-fatal — fall through with first-pass output
  }

  // Step 5: parse_to_dict_safely and return (mirrors lines 260-261)
  return parseToDictSafely(rawAnalystText)
}
