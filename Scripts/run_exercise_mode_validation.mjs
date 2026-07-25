// Faithful replica of squat-frontend/src/analyzeService.js's analyzeLift(),
// but reading pre-extracted frames from disk (via extract_exercise_mode_frames.py)
// instead of a browser File/Video/Canvas — same client, same model, same exact
// prompts/exemplar, same two-pass (analyst + reflection) flow. This is for
// offline validation against the real Reddit r/formcheck clips, not a
// reimplementation of the logic — every string below is copy-identical to
// analyzeService.js.
import fs from 'fs'
import path from 'path'
import { GoogleGenAI } from '@google/genai'

const apiKey = process.env.GEMINI_API_KEY
if (!apiKey) {
  console.error('Set GEMINI_API_KEY in your environment before running this script (same key analyzeService.js uses).')
  process.exit(1)
}
const client = new GoogleGenAI({ apiKey })
const ANALYST_MODEL_ID = 'models/gemini-3.1-flash-lite-preview'

const BASE_SYSTEM_INSTRUCTION = `You are an Expert Biomechanical Analyst specializing in powerlifting squat analysis.
You will be provided with a chronological grid of visual frames spanning an entire training set.
Analyze the global movement patterns, count the exact repetition scope across the whole video, and output a precise form rating based on our official rubric.
`

const COT_INSTRUCTION_APPEND = `
### 🧠 STRUCTURAL CHAIN-OF-THOUGHT PROTOCOL:
Before building the final JSON payload, you MUST perform a critical validation review.
Evaluate each deduction item based on the full timeline sequence. If an error is a minor stabilization anomaly, do not penalize.
`

const SYSTEM_INSTRUCTION = BASE_SYSTEM_INSTRUCTION + COT_INSTRUCTION_APPEND

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

const TASK_INSTRUCTION = "### 🎯 CURRENT RUN ASSESSMENT TARGET:\nEvaluate this comprehensive frame matrix timeline sequence blindly. Output raw JSON."

function buildReflectionPrompt(rawDraftText) {
  return `You are an Expert Biomechanical Verification Auditor.
Review your draft squat assessment carefully to eliminate false positives or alignment anomalies.
Ensure your parsed repetition count matches the full frame matrix timeline scope.

[DRAFT ASSESSMENT]:
${rawDraftText}

Return a revised, structurally pristine JSON block following the identical schema rules. Output ONLY JSON.`
}

function parseToDictSafely(textPayload) {
  try {
    let clean = textPayload.trim()
    clean = clean.replace(/^```(?:json)?/i, '').replace(/```$/, '').trim()
    return JSON.parse(clean)
  } catch {
    return { parsing_error: true, raw_text: textPayload }
  }
}

function frameToInlinePart(jpegPath) {
  const data = fs.readFileSync(jpegPath).toString('base64')
  return { inlineData: { mimeType: 'image/jpeg', data } }
}

async function analyzeLiftFromFrames(frameDir, sam3Enabled = false) {
  const targetFrames = fs.readdirSync(frameDir)
    .filter(f => f.endsWith('.jpg'))
    .sort()
    .map(f => path.join(frameDir, f))

  if (targetFrames.length === 0) {
    throw new Error(`No frames found in ${frameDir}`)
  }

  const parts = [
    { text: SYSTEM_INSTRUCTION },
    {
      text:
        `=== EXEMPLAR 1 ANSWER METRICS ===\n` +
        `Ground-Truth JSON:\n${JSON.stringify(EXEMPLAR_1_JSON, null, 2)}\n` +
        `Text Feedback:\n${EXEMPLAR_1_TEXT}\n`,
    },
    ...targetFrames.map(frameToInlinePart),
    ...(sam3Enabled
      ? [{ text: '[SAM3 MODE] This video has been pre-processed with skeleton keypoint overlays. Weight joint-angle evidence more heavily than raw pixel appearance when assessing depth and alignment.' }]
      : []),
    { text: TASK_INSTRUCTION },
  ]

  const firstResp = await client.models.generateContent({
    model: ANALYST_MODEL_ID,
    contents: [{ role: 'user', parts }],
  })
  let rawAnalystText = firstResp.candidates?.[0]?.content?.parts?.[0]?.text ?? ''

  try {
    const reflectionResp = await client.models.generateContent({
      model: ANALYST_MODEL_ID,
      contents: [{ role: 'user', parts: [{ text: buildReflectionPrompt(rawAnalystText) }] }],
    })
    const reflectedText = reflectionResp.candidates?.[0]?.content?.parts?.[0]?.text ?? ''
    if (reflectedText) rawAnalystText = reflectedText
  } catch {
    // non-fatal, same as production
  }

  return parseToDictSafely(rawAnalystText)
}

const framesRoot = process.argv[2]
const outputJson = process.argv[3]

analyzeLiftFromFrames(framesRoot)
  .then(result => {
    fs.writeFileSync(outputJson, JSON.stringify(result, null, 2))
    console.log(`OK -> ${outputJson}`)
    console.log(JSON.stringify(result, null, 2))
  })
  .catch(err => {
    console.error('FAILED:', err.message)
    process.exit(1)
  })
