// Turns raw rehab pipeline numbers into plain-language, color-coded
// interpretations a patient (not just a clinician) can act on.

const GOOD = '#56c596'
const INFO = '#7eb8d4'
const WARN = '#e5a14a'
const BAD = '#e06c75'

// camera_alignment comes pre-classified from the backend (CameraAlignmentChecker
// in rehab_knee_extension.py) — this just maps its level to display color/label.
// The thresholds themselves were derived empirically (measured against a known-
// good side-on clip vs. several frontal clips), not from a clinical source —
// see the backend docstring for the actual numbers and reasoning.
export function classifyCameraAlignment(cameraAlignment) {
  if (!cameraAlignment || cameraAlignment.level === 'unknown') {
    return null
  }
  if (cameraAlignment.level === 'good') {
    return { label: 'Good Camera Angle', color: GOOD, note: null, cue: null }
  }
  if (cameraAlignment.level === 'caution') {
    return { label: 'Camera Angle Slightly Off', color: WARN, note: cameraAlignment.message, cue: 'Adjust your camera a little — get more side-on.' }
  }
  return { label: 'Camera Not Side-On', color: BAD, note: cameraAlignment.message, cue: 'Turn your camera fully side-on before continuing.' }
}

// trunk_compliance comes pre-classified from the backend (TrunkComplianceChecker)
// — flags compensatory trunk lean/rocking, one of the most commonly cited form
// faults for this exact exercise (multiple real PT demo videos used to test this
// project explicitly call out "Leaning Backward" and "Using Momentum" as mistakes
// to avoid). The degree thresholds are an empirical derivation (measured against
// a clean controlled clip's own natural sway), not a clinical citation — stated
// here rather than implied, same as the camera-alignment check.
export function classifyTrunkCompliance(trunkCompliance) {
  if (!trunkCompliance || trunkCompliance.level === 'unknown') {
    return null
  }
  if (trunkCompliance.level === 'good') {
    return { label: 'Stable Trunk', color: GOOD, note: null, cue: null }
  }
  if (trunkCompliance.level === 'caution') {
    return { label: 'Slight Trunk Sway', color: WARN, note: trunkCompliance.message, cue: 'Sit a little taller — keep your trunk still.' }
  }
  return { label: 'Compensatory Trunk Lean', color: BAD, note: trunkCompliance.message, cue: 'Stop leaning your back — sit tall and isolate the knee.' }
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
      note: 'Your leg was lost in some frames — results below are still usable but may be a little noisy.',
    }
  }
  return {
    label: 'Poor Tracking', color: BAD, pct,
    note: 'Your leg was hard to track for most of this clip. Try better lighting and make sure your hip, knee, and ankle are all clearly visible from a side angle, then re-record before trusting these numbers.',
  }
}

export function classifyExtensionDeficit(deficitDeg) {
  const deg = Math.round(deficitDeg * 10) / 10
  if (deg <= 5) {
    return {
      label: 'Full Extension', value: deg, color: GOOD,
      note: 'Your knee reaches a fully straight position — within the normal range.',
      cue: null,
    }
  }
  if (deg <= 10) {
    return {
      label: 'Mild Extension Deficit', value: deg, color: INFO,
      note: 'Your knee falls a little short of fully straight. This is common early in recovery and usually improves with continued reps.',
      cue: 'Push a little further toward full extension.',
    }
  }
  if (deg <= 20) {
    return {
      label: 'Moderate Extension Deficit', value: deg, color: WARN,
      note: 'Your knee is noticeably short of full extension. Worth mentioning to your physical therapist.',
      cue: "Extend further — you're stopping short of straight.",
    }
  }
  return {
    label: 'Significant Extension Deficit', value: deg, color: BAD,
    note: 'Your knee is far from fully straight on this rep. Share this clip with your physical therapist.',
    cue: "Go up a lot further — you're well short of full extension.",
  }
}

// NOTE: seated knee extension is one of the exercises where physical
// therapists specifically instruct patients to move SLOWLY and under
// control ("don't kick it up fast, don't swing the leg") — so unlike a lot
// of other rehab movements, there's no clinically-correct "fast = healthy"
// speed range here. A slow, careful, correct rep and a slow, pain-guarded
// rep can land at the exact same velocity. So a LOW number is never graded
// bad on its own — the pattern of the motion (see classifyMuscleActivation)
// carries that weight instead.
//
// A genuinely HIGH number is a different story: multiple PT sources
// consistently warn against using momentum/swinging the leg for this exact
// exercise family —
//   https://www.zing.coach/fitness-library/quad-exercises-for-knee-pain
//   https://wrightpt.com/knee-rehab-exercises/
//   https://www.medicalnewstoday.com/articles/325804
// — but none of them give a specific deg/s cutoff for THIS exercise. The
// BALLISTIC_VELOCITY_CEILING below is derived, not found: ACSM's general
// resistance-training tempo guidance recommends roughly a 1-2s concentric
// (lifting) phase for novice/intermediate lifters —
//   https://pmc.ncbi.nlm.nih.gov/articles/PMC8310485/
//   https://www.ideafit.com/progression-models-in-resistance-training-for-healthy-adults/
// — which for this exercise's ~90deg range works out to roughly 45-90deg/s
// average. A smooth rep's INSTANTANEOUS peak velocity typically runs
// 1.5-2x its average, putting a controlled-but-brisk rep's peak around
// 90-180deg/s. 200deg/s is set comfortably above that band specifically to
// avoid flagging a merely-brisk-but-controlled rep — it's a proxy for
// "exceeds the general controlled-tempo envelope," not a clinically
// validated injury threshold for this specific exercise, and that
// distinction is worth saying out loud if asked.
const BALLISTIC_VELOCITY_CEILING = 200

export function classifyMovementSpeed(velocityDegS) {
  const v = Math.round(velocityDegS * 10) / 10
  if (v > BALLISTIC_VELOCITY_CEILING) {
    return {
      label: 'Possible Momentum / Ballistic Movement', value: v, color: BAD,
      note: `This rep's peak speed (${v}°/s) is well above the range a controlled, deliberate rep should produce. Physical therapy guidance for this exercise consistently warns against swinging the leg or using momentum — slow down and let your quad do the work, not gravity or a kicking motion.`,
      source: 'https://www.medicalnewstoday.com/articles/325804',
      cue: "Slow down — don't swing or kick your leg up.",
    }
  }
  return {
    label: 'Movement Speed', value: v, color: INFO,
    note: 'Shown for reference only. Unlike many exercises, slow and controlled is the clinically correct way to do a seated knee extension, so a low number here is not, by itself, a warning sign.',
    cue: null,
  }
}

// eccentric_duration_s/is_descent_too_fast come from the backend's
// ExtensionDeficitTracker (rehab_knee_extension.py) — it times the lowering
// phase of each rep (peak extension -> back to the bent starting position).
// The ~2s floor mirrors the same ACSM-derived tempo reasoning documented
// next to BALLISTIC_VELOCITY_CEILING above, applied to the descent instead
// of the ascent: PT guidance for this exercise says control the drop, don't
// let gravity do it.
export function classifyEccentricPacing(eccentricDurationS, isDescentTooFast) {
  if (eccentricDurationS === null || eccentricDurationS === undefined) {
    return null
  }
  const s = Math.round(eccentricDurationS * 10) / 10
  if (isDescentTooFast) {
    return {
      label: 'Fast Descent', value: `${s}s`, color: BAD,
      note: `This rep's lowering phase took about ${s}s — quicker than the controlled pace this exercise calls for. Letting the leg drop instead of lowering it under control skips the muscle work the eccentric phase is meant to train.`,
      cue: 'Slow down on the way down — control the descent.',
    }
  }
  return {
    label: 'Controlled Descent', value: `${s}s`, color: GOOD,
    note: 'Your lowering phase was paced well — under control rather than dropped.',
    cue: null,
  }
}

export function classifyMuscleActivation(isInhibited) {
  if (!isInhibited) {
    return {
      label: 'Smooth Activation', color: GOOD,
      note: 'No stop-start hesitation detected — your muscle fired in one continuous motion, whether quickly or slowly.',
      cue: null,
    }
  }
  return {
    label: 'Possible Muscle Inhibition', color: BAD,
    note: "Your leg's movement paused and restarted partway through the rep, instead of moving in one continuous motion — this stutter pattern shows up regardless of how fast or slow the rep was. It can happen with pain or muscle guarding after an injury or surgery, but it isn't a diagnosis on its own. If this surprises you, mention the clip to your physical therapist.",
    cue: 'Try to move in one smooth motion, without pausing partway.',
  }
}

// Looks at the WHOLE set of reps in a clip, not any single one — this is
// the thing a "best rep" summary structurally cannot tell you: is the
// patient getting better or worse within this set? Compares the average
// extension deficit across the first half of the set vs. the second half.
// A real, steady trend in either direction matters more clinically than the
// raw deficit number on any one rep; high rep-to-rep variance with no clear
// direction is itself a separate, useful signal (inconsistent effort or
// camera drift), so it's called out distinctly rather than averaged away.
export function classifyRepTrend(reps) {
  if (!reps || reps.length < 2) {
    return { label: 'Not Enough Reps', color: INFO, deltaDeg: 0, note: 'Record at least 2 reps to see a within-set trend.', cue: null }
  }

  const deficits = reps.map(r => r.extension_deficit_deg)
  const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length
  const mid = Math.ceil(deficits.length / 2)
  const firstHalfAvg = mean(deficits.slice(0, mid))
  const secondHalfAvg = mean(deficits.slice(mid))
  const delta = Math.round((secondHalfAvg - firstHalfAvg) * 10) / 10 // positive = deficit grew = got worse

  const overallMean = mean(deficits)
  const stdDev = Math.sqrt(mean(deficits.map(d => (d - overallMean) ** 2)))

  // High scatter with no consistent direction is reported as its own
  // category rather than washed out by the half-vs-half average.
  if (stdDev > 8 && Math.abs(delta) < 5) {
    return {
      label: 'Inconsistent', color: WARN, deltaDeg: delta,
      note: 'Your extension deficit bounced around from rep to rep rather than trending steadily in either direction — this can mean uneven effort between reps, or the camera/leg position shifting slightly during the set.',
      cue: 'Try to keep your effort and position steady rep to rep.',
    }
  }
  if (delta > 5) {
    return {
      label: 'Fatiguing', color: BAD, deltaDeg: delta,
      note: `Your extension deficit grew by about ${Math.abs(delta)}° from the first half of the set to the second half — your range of motion shrank as you went, a common sign of fatigue. Consider fewer reps per set or more rest between sets.`,
      cue: "You're tiring — consider resting or stopping the set soon.",
    }
  }
  if (delta < -5) {
    return {
      label: 'Improving Within Set', color: GOOD, deltaDeg: delta,
      note: `Your extension deficit shrank by about ${Math.abs(delta)}° from the first half of the set to the second half — you're loosening up or warming into the movement as you go.`,
      cue: null,
    }
  }
  return {
    label: 'Consistent', color: GOOD, deltaDeg: delta,
    note: 'Your range of motion stayed steady from rep to rep — no signs of fatigue or warm-up drift in this set.',
    cue: null,
  }
}

// Drives the live ROM gauge during an active recording. Deliberately generic
// (angle + a [min, max] target range, both supplied by the backend's
// /live-status payload) rather than knee-extension-specific, so a future
// movement (e.g. sit-to-stand) can reuse this same gauge component just by
// reporting its own target_range — no gauge-side changes needed.
export function classifyLiveRom(angle, targetRange) {
  if (angle === null || angle === undefined || !targetRange) {
    return { pct: 0, color: INFO, label: '—' }
  }
  const [lo, hi] = targetRange
  const pct = Math.max(0, Math.min(1, (angle - lo) / (hi - lo)))
  let color = BAD
  if (pct >= 0.9) color = GOOD
  else if (pct >= 0.6) color = WARN
  return { pct, color, label: `${Math.round(angle)}°` }
}

export function classifyLSI(pct) {
  if (pct >= 90) {
    return {
      label: 'Balanced', color: GOOD,
      note: 'Both legs are performing similarly — a strong sign of balanced recovery.',
    }
  }
  if (pct >= 75) {
    return {
      label: 'Mild Asymmetry', color: WARN,
      note: 'One leg is noticeably behind the other. Worth tracking over your next few sessions.',
    }
  }
  return {
    label: 'Significant Asymmetry', color: BAD,
    note: 'One leg is performing well below the other. Recommend discussing this with your physical therapist before progressing load.',
  }
}
