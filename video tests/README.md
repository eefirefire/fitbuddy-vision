# Video Tests

Real, unscripted video clips used to regression-test the live rehab
pipelines (`Scripts/rehab_knee_extension.py` and `Scripts/
rehab_sit_to_stand.py`) end-to-end — real MediaPipe pose detection, real
human movement, run through the actual pipeline code instead of synthetic
angle arrays. Built while debugging the Tier 1 live rep-counter/voice-cue
feature; kept here so the same scenarios can be re-checked after future
changes without re-hunting for source clips.

### `sit_to_stand_test.mp4`
Source: "30-Second Sit-to-Stand Test | Muscle Power Assessment in Elderly"
(YouTube, ~4min, downloaded via yt-dlp for local testing only). Run through
`rehab_sit_to_stand.run_batch_validation()` directly (not the live UI —
see that module):
- 14 real reps detected across the full clip with zero crashes, no NaN
  propagation, no negative deficits.
- Hip and knee angles both extend together and land in a plausible range
  (165-180 deg) for genuine full stands.
- Two reps (angles ~90-100 deg) were correctly flagged as genuinely short
  rather than silently averaged in with the rest — likely a different demo
  segment in the source video, not a tracking failure.
- **Calibration caveat, not a bug:** this source video is the timed
  30-second MAX-REPS test, where fast/repeated motion is the *correct*
  behavior — so most reps got flagged `is_descent_too_fast` against this
  module's pacing thresholds. Those thresholds are intentionally tuned for
  the OEP-style CONTROLLED single rep (the actual clinical context this
  feature was built for), not the timed test. This video was used only to
  validate the tracking geometry against real motion, not to tune pacing.

### `sit_to_stand_controlled.mp4`
Source: "Sit to Stand Strengthening Exercise" by Signature Medical Group /
Dave Reddy (YouTube, ~4.5min, downloaded via yt-dlp for local testing only).
A trainer demoing controlled reps with narration between them — closer to
the actual OEP-style controlled pacing this module's thresholds are tuned
for than the timed max-reps test above, and the clip that found three real
bugs in `rehab_sit_to_stand.py` (see "Bugs found and fixed" below). Run
through `run_batch_validation()` after those fixes:
- 6 "reps" detected. Rep 1 (standing_angle 172.1) is a clean genuine stand.
- **Reps 2-4 are real, not a bug — just an artifact of this specific demo,
  verified by inspecting the raw per-frame angle trace directly, not just
  the summary numbers.** The trainer stands and talks for extended periods
  (13-97s) between reps rather than sitting straight back down; the long
  `eccentric_duration_s` values (97.0, 34.5, 13.28) are genuinely measuring
  "time spent still elevated before sitting low enough to re-arm," not a
  slow/bad descent, and `is_jerky` is genuinely true — the trainer's
  natural weight-shifting and gesturing while standing still really does
  produce non-smooth motion in the tracked angle, even though it isn't
  clinically meaningful. Voice-cue behavior was hand-verified frame-by-frame
  against this: `Good` fires correctly on rep 1, `Slow down` fires exactly
  once per jerky rep (correctly suppressing `Good`/`Stand up further` on
  each), and nothing fires falsely. A real single-patient session won't
  have multi-minute narration gaps like this, so this pattern is unlikely
  to recur Sunday, but if Dad pauses and talks for a long time mid-session,
  expect a similar "technically correct, clinically confusing" report on
  that rep — not a crash, not wrong data, just a metric that isn't
  meaningful outside its intended controlled-single-rep context.
- Reps 5-6 (standing_angle ~43-50 deg) are clearly not full stands — most
  likely a different exercise demonstrated later in the same video, not a
  tracking failure (both hip and knee angles agree with each other at that
  low angle, and the reading is stable across frames, not noisy).
- Zero crashes, no NaN propagation, across the full ~275s clip.

### `sit_to_stand_walker.mp4`
Source: "How to Sit Down and Stand Up with a Walker" by RegisteredNurseRN
(YouTube, ~77s, downloaded via yt-dlp for local testing only). A nursing
demo of the walker-assisted sit-to-stand transfer — relevant because
walker use is common in this app's actual target population (older
adults with fall risk) and wasn't tested before. Run through
`run_batch_validation()` after the fixes below:
- 1 rep detected (standing_angle 167.3, peak velocity 26.9 deg/s — a
  genuinely slow, controlled, instructional-pace stand). `Good` fires
  correctly, no false cues. The earlier run (before the settling/glitch
  fixes) showed a spurious extra "rep" at 52.8 degrees very early in the
  clip; re-inspection confirmed that was settling-window noise, not a
  second real segment as first guessed — it's gone now that settling uses
  a robust (median) baseline instead of a bare running minimum.
- No crash or landmark failure from the walker being held in both hands —
  pose detection tracked hip/knee/shoulder normally despite the
  assistive device. This is a genuinely useful data point for Sunday if
  Dad uses any kind of support.

### Bugs found and fixed in `rehab_sit_to_stand.py` this pass
Found by testing cue correctness against real footage, not just checking
for crashes — three real, verified bugs, each with a regression test:
1. **A MediaPipe tracking glitch could permanently corrupt rep detection.**
   `sit_to_stand_controlled.mp4` has one frame where the composite angle
   jumps ~115 degrees in a single frame (~3800 deg/s, physically
   impossible) while MediaPipe reported normal-to-high landmark confidence
   throughout — so confidence-score gating alone doesn't catch it. Fixed
   with two independent guards in `_CandidateBodySideTrack.push_frame`: a
   raw-velocity ceiling for single-frame teleports, and a
   floor-relative-to-calibrated-baseline check for gradual multi-frame
   drift, since the glitch's own internal per-frame deltas were each too
   small to trip a velocity check alone.
2. **Batch validation (`run_batch_validation`, the function this Sunday's
   goniometer comparison will use) never calibrated a settling baseline at
   all** — live sessions get 3 seconds of setup time before tracking
   starts; batch mode went straight from frame 0, so a video's opening
   frames (camera panning in, subject not yet framed) could anchor the
   entire session's rep-detection floor. Fixed by adding a short (0.5s)
   settling window to batch mode too.
3. **The settling window itself used a running MINIMUM, the least
   robust statistic possible against contaminated calibration frames** — a
   single bad opening frame would poison the whole session. Fixed by
   collecting all settling-window samples and committing the calibrated
   baseline as their MEDIAN once settling ends, both for live sessions and
   batch validation.

All three are covered by new tests in `test_rehab_sit_to_stand.py`
(42 passing total, up from 18). Also separately fixed (session-quality, not
correctness): the AMI/jerk classifier was firing during genuinely
motionless rest periods due to landmark micro-jitter alone — gated behind a
minimum real-motion floor so only actual physical movement can trigger it.

**Update:** `seated_knee_extension_real.mp4` (added later) IS real footage of
this exact exercise, properly side-on — use that one first. The squat clips
below were stand-ins used before a real clip was available; they're kept
because they still exercise the mechanics fine (rep detection, cue
triggering, state transitions — the angle math is exercise-agnostic), but
`camera_alignment` will correctly read `"poor"` on all of them since none are
a proper side-on shot, and their peak-angle/deficit numbers aren't clinically
meaningful. That's expected on those three, not a bug.

Before trusting any real clinical validation beyond spot-checking, retest
against footage of an actual patient/volunteer on the real target hardware.

## Voice-cue coverage (all 4, on real seated knee-extension footage)

| Cue | Verified on | How |
|---|---|---|
| "Good" | `seated_knee_extension_real.mp4` | Live pipeline + voice, 2 independent runs |
| "Slow down" | both real seated-extension clips | Live pipeline + voice, both AMI-stutter and fast-descent triggers |
| "Check your camera angle" | `seated_knee_extension_band_vertical.mp4` | Live pipeline + voice, fired on 2 genuine alignment transitions (not spam) |
| "Extend further" | `seated_knee_extension_band_vertical.mp4` | Live pipeline + voice, genuinely spoken twice in one real session (see below) |

**Fixed:** cues used to live in a single-slot `{text, seq}` per channel, so
when a rep earned two DIFFERENT cues close together (e.g. "Extend further"
at peak-commit, then "Slow down" moments later once the descent was
measured), the second silently overwrote the first before the frontend's
400ms poll ever saw it — "Extend further" fired correctly on the backend
but was never actually heard. `CueChannel` now keeps a short bounded
history and exposes `cue_pending`/`alignment_cue_pending` (everything since
the client's last-seen seq, oldest first) alongside the existing single
"current" value; the frontend drains and speaks all of them instead of only
the latest. Verified end-to-end: a real session on this clip produced the
literal spoken sequence `Extend further → Slow down → Good → Slow down →
Check your camera angle → Slow down → Good → Extend further`, both
"Extend further" occurrences included.

## How to run one

Point the backend at a clip instead of a live webcam:

```bash
cd Scripts
REHAB_VIDEO_SOURCE="../video tests/<file>.mp4" python rehab_knee_extension.py
```

Then either open the app normally (Rehab Mode → Live Camera → Start Camera,
after clicking "Next" into a RECORD state) — the `<img>` stream will read
from the file instead of a webcam — or drive it directly via the API
(register → intake → advance → open `/api/rehab/stream` → poll
`/api/rehab/live-status`).

**Windows tip:** always confirm nothing is already bound to port 5050 before
starting (`Get-NetTCPConnection -LocalPort 5050`) — Flask's debug reloader
can leave a stray child process running if a previous run wasn't stopped
cleanly, which causes exactly the kind of inconsistent/"random" behavior
this suite was built to catch.

## Clips

### `seated_knee_extension_real.mp4`
Source: "Seated Knee Extension (LAQ)" by Ask Doctor Jo (YouTube, ~26s,
downloaded via yt-dlp for local testing only — not redistributed). This
channel is already cited as reference material in this module's own code
comments (see TrunkComplianceChecker), so it's a good match for what the
pipeline was actually tuned against. **The best fixture in this folder** —
genuine side-on seated knee extension, not a stand-in.
Confirms (verified via real UI clicks, not just API calls):
- Rep counter correctly detects 3 real reps.
- **`camera_alignment` correctly reads `"good"`** — the first fixture where
  this could actually be tested, since the squat clips below are all
  structurally off-angle. No alignment cue false-fired the entire session
  (`alignment_cue_seq` stayed 0).
- State-transition voice hint fires on a real "Next" click ("Perform your
  seated knee extension reps now.").
- Rep-quality voice cues fire in a sensible sequence tied to real reps
  ("Slow down" → "Good" → "Slow down"), no duplicates/spam.
- Full session flow works end-to-end through real button clicks: Next →
  Live Camera → Start Camera → (reps happen) → Stop Camera → coherent
  ClipSummary with real, specific feedback (trunk sway, descent pacing).
- Zero browser console errors across the whole flow.

### `seated_knee_extension_band_vertical.mp4`
Source: "Seated Knee Extension Exercise with Resistance Band" by LifeStrength
Physical Therapy (YouTube Shorts, ~50s, downloaded via yt-dlp for local
testing only). Filmed vertically/frontally on a phone (typical for a
Shorts-format clip) — genuinely off-angle, unlike the clip above.
Confirms (verified via real UI clicks + voice interception):
- **`camera_alignment` correctly reads `"poor"`** on real footage of this
  exact exercise (ratio 0.649, well past the poor threshold).
- **"Check your camera angle"** cue fires audibly through the real live
  pipeline, twice, on genuine transitions as the person's position shifted
  during the exercise (~13s apart — not spam).
- Rep counter and "Good"/"Slow down" cues continue working correctly
  alongside the alignment cue firing (4 reps counted, no cross-channel
  interference).

### `rep_counter_good_and_slowdown.mp4`
Source: a real, ~49s multi-rep squat clip with clean, controlled reps.
Confirms:
- Rep counter increments correctly across multiple real reps (0 → 1 → 2 → …).
- **"Good"** cue fires on a clean rep.
- **"Slow down"** cue fires (verified via both triggers: mid-rep stutter via
  the AMI classifier, and a fast/dropped descent).
- No duplicate or repeated cues for the same event — `cue_seq` advances
  exactly once per real state change.
- Camera-alignment cue ("Check your camera angle") fires exactly once on
  the transition into "poor", not on every frame it stays poor.

### `extend_further_short_rep.mp4`
Source: a real failed heavy-squat clip — the lifter never comes close to
standing up straight, a single clean (non-jittery) short rep.
Confirms:
- **"Extend further"** fires correctly on a rep that falls short of the
  extension target but isn't otherwise flagged (not inhibited).
- Note: in this specific clip, the same rep also drops fast afterward, so
  "Slow down" supersedes "Extend further" moments later — that's the
  cue-priority system working as intended (a rep with multiple faults shows
  the higher-priority one), not a bug. If you need "Extend further" to be
  the *final* cue on screen, you'd need a clip with a controlled (not fast)
  descent after a short rep.

### `older_adult_smoke_test.mp4`
Source: a real ~10s clip of a 68-year-old lifter. Useful as a general smoke
test / thematically closer to this app's actual target audience than the
other two. No specific cue guaranteed — good for a quick "does the pipeline
still run end-to-end without errors" check after a change.
