// Turns raw sit-to-stand pipeline numbers into plain-language, color-coded
// interpretations — sibling to rehabInterpret.js, reusing what's genuinely
// generic from it (camera alignment, eccentric pacing, the live ROM gauge)
// rather than re-deriving them, and adding only what sit-to-stand actually
// needs on top of that.
import { classifyCameraAlignment, classifyEccentricPacing, classifyLiveRom } from './rehabInterpret'

export { classifyCameraAlignment, classifyEccentricPacing, classifyLiveRom }

const GOOD = '#56c596'
const INFO = '#7eb8d4'
const WARN = '#e5a14a'
const BAD = '#e06c75'

// standing_deficit_deg = IDEAL_STANDING_DEG (165°, see StandTransitionTracker
// in the backend) minus the peak composite (min of hip/knee) angle reached.
// Same "honesty about validation status" caveat the backend states
// explicitly applies here: these bands are a product choice for readable
// feedback, not an independently validated clinical cutoff for sit-to-stand
// specifically — unlike the knee-extension deficit bands, which at least
// share the same underlying formula the goniometer validation was run
// against.
export function classifyStandingDeficit(deficitDeg) {
  const deg = Math.round(deficitDeg * 10) / 10
  if (deg <= 5) {
    return {
      label: 'Full Stand', value: deg, color: GOOD,
      note: 'You reached a fully upright position on this rep.',
      cue: null,
    }
  }
  if (deg <= 15) {
    return {
      label: 'Mild Standing Deficit', value: deg, color: INFO,
      note: "You're a little short of fully upright. Common and usually improves with continued reps.",
      cue: 'Try to stand up a little further.',
    }
  }
  if (deg <= 30) {
    return {
      label: 'Moderate Standing Deficit', value: deg, color: WARN,
      note: 'Noticeably short of a full stand on this rep.',
      cue: "Stand up further — you're stopping short.",
    }
  }
  return {
    label: 'Significant Standing Deficit', value: deg, color: BAD,
    note: 'Far short of a full stand on this rep — worth mentioning to a physical therapist.',
    cue: "Push further up — you're well short of standing.",
  }
}

// is_jerky comes from the backend's AMIAnomalyClassifier, reused generically
// on the composite standing-angle signal — see that class's docstring for
// why this is presented as a purely descriptive "the rise wasn't smooth"
// flag, not a clinical diagnosis (a stuttering stand could reflect balance
// hesitation, weakness, or pain; this pipeline can't and doesn't try to
// distinguish between those).
export function classifyJerkyMotion(isJerky) {
  if (!isJerky) {
    return {
      label: 'Smooth Rise', color: GOOD,
      note: 'No stop-start hesitation detected — the stand happened in one continuous motion.',
      cue: null,
    }
  }
  return {
    label: 'Uneven Rise', color: BAD,
    note: "The rise paused and restarted partway through instead of moving in one continuous motion. This isn't a diagnosis on its own — it can reflect balance hesitation, weakness, or pain — but is worth mentioning if it surprises you.",
    cue: 'Try to stand up in one smooth motion, without pausing partway.',
  }
}

export function classifyTrackingQuality(framesWithPose, framesProcessed) {
  if (!framesProcessed) {
    return { label: 'No Data', color: WARN, pct: 0, note: 'No frames were processed from this clip.' }
  }
  const pct = Math.round((framesWithPose / framesProcessed) * 100)
  if (pct >= 90) {
    return { label: 'Good Tracking', color: GOOD, pct, note: null }
  }
  if (pct >= 70) {
    return {
      label: 'Fair Tracking', color: WARN, pct,
      note: 'You were lost in some frames — results below are still usable but may be a little noisy.',
    }
  }
  return {
    label: 'Poor Tracking', color: BAD, pct,
    note: 'You were hard to track for most of this clip. Try better lighting and make sure your hip, knee, and ankle are all clearly visible from a side angle, then re-record before trusting these numbers.',
  }
}
