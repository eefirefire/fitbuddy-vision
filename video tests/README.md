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
for than the timed max-reps test above. Run through `run_batch_validation()`:
- 6 "reps" detected. Reps 1-4 (standing_angle 155-172 deg) are plausible
  genuine stands with correctly-clamped small deficits.
- **Real limitation surfaced, not a crash:** reps 2-4 show implausibly long
  `eccentric_duration_s` (97s, 34.5s, 13.28s) and got flagged `is_jerky`.
  This is the trainer talking/gesturing between demo reps — small sustained
  motion during a long pause gets picked up as part of the "descent" instead
  of a new settled baseline, so the duration and jerk numbers for those reps
  aren't clinically meaningful. A real single-patient session (stand, sit,
  brief pause, repeat) won't have multi-minute narration gaps like this, but
  it's a real edge case worth knowing about: if Dad talks or fidgets for a
  long time mid-session, expect a noisy report on that rep, not a crash.
- Reps 5-6 (standing_angle ~43-50 deg) are clearly not full stands — most
  likely a different exercise demonstrated later in the same video, not a
  tracking failure (both hip and knee angles agree with each other at that
  low angle, which is what you'd expect from a real seated/bent position,
  not sensor noise).
- Zero crashes, no NaN propagation, across the full ~275s clip.

### `sit_to_stand_walker.mp4`
Source: "How to Sit Down and Stand Up with a Walker" by RegisteredNurseRN
(YouTube, ~77s, downloaded via yt-dlp for local testing only). A nursing
demo of the walker-assisted sit-to-stand transfer — relevant because
walker use is common in this app's actual target population (older
adults with fall risk) and wasn't tested before. Run through
`run_batch_validation()`:
- 2 reps detected. Rep 2 (standing_angle 167.3 deg) is a clean, plausible
  full stand.
- Rep 1 (standing_angle 52.8 deg, both hip and knee angles agreeing at
  that low position) is real footage but not a full stand — likely an
  earlier segment of the demo (seated/mid-transfer position), same
  "different segment, not sensor noise" pattern as above.
- No crash or landmark failure from the walker being held in both hands —
  pose detection tracked hip/knee/shoulder normally despite the
  assistive device. This is a genuinely useful data point for Sunday if
  Dad uses any kind of support.

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
