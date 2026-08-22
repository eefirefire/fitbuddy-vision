# Video Tests

Real, unscripted video clips used to regression-test the live rehab
pipelines (`Scripts/rehab_knee_extension.py` and `Scripts/
rehab_sit_to_stand.py`) end-to-end — real MediaPipe pose detection, real
human movement, run through the actual pipeline code instead of synthetic
angle arrays. Built while debugging the Tier 1 live rep-counter/voice-cue
feature; kept here so the same scenarios can be re-checked after future
changes without re-hunting for source clips.

## Sit-to-stand

`rehab_sit_to_stand.py` went through three rounds of code review plus a
brief removal/restoration (see git history around commits `ab592a9` through
`48c3974` and this file's history for the blow-by-blow). Final state, after
closing the last two known concurrency gaps:

- **Two frame sources (live stream vs. `/upload`/batch) can no longer
  corrupt each other's settling calibration** even if both somehow fed the
  same shared session — `route_frame` now takes a `source` tag and ignores
  any frame from a source that didn't claim the session first, since the
  two sources use incompatible timestamp bases (wall-clock vs.
  video-relative).
- **`/upload` now holds `_session_lock` for the WHOLE request**, not just
  per-frame — a concurrent `/advance` can no longer land between two
  frames of a long upload and reset tracking state mid-video. This blocks
  other requests for the duration of an upload, which is the correct
  tradeoff for this module's single-shared-session design (nothing should
  be mutating session state while a user-initiated upload is still being
  processed).

59 tests passing. Re-verified against both real test videos below with
these fixes in place — results are identical to the prior (already-hardened)
run, confirming these last two fixes closed real gaps without changing any
actual tracking behavior:

### `sit_to_stand_controlled.mp4`
Source: "Sit to Stand Strengthening Exercise" by Signature Medical Group /
Dave Reddy (YouTube, ~4.5min, downloaded via yt-dlp for local testing only).
A trainer demoing controlled reps with narration between them. Run through
`run_batch_validation()`:
- 6 reps detected, 6 rejected frames (real MediaPipe tracking glitches
  correctly caught and excluded, not silently corrupting the session).
  Rep 1 (172.1°) is a clean genuine stand. Reps 2-4 have long
  `eccentric_duration_s` values (97s, 34.5s, 13.28s) and are flagged
  jerky — confirmed via direct frame-trace inspection to be real: the
  trainer stands and talks between reps rather than sitting back down,
  and the natural weight-shifting genuinely isn't smooth motion, even
  though it isn't clinically meaningful. Reps 5-6 (~43-50°) are a
  different segment of the source video, not full stands.
- Voice-cue behavior hand-verified frame-by-frame: `Good` fires correctly
  on rep 1, `Slow down` fires exactly once per jerky rep, nothing fires
  falsely.
- Zero crashes, no NaN propagation, across the full ~275s clip.

### `sit_to_stand_walker.mp4`
Source: "How to Sit Down and Stand Up with a Walker" by RegisteredNurseRN
(YouTube, ~77s, downloaded via yt-dlp for local testing only). Relevant
because walker use is common in this app's actual target population.
- 2 reps detected, 0 rejected frames. Rep 2 (167.3°, peak velocity
  26.9 deg/s) is a clean, genuinely slow/controlled stand — `Good` fires
  correctly. Rep 1 (52.8°) is a real but non-representative early segment
  of the source demo (a mid-transfer or seated position caught before the
  subject is fully framed), not sensor noise.
- No crash or landmark failure from the walker being held in both hands —
  pose detection tracked hip/knee/shoulder normally despite the assistive
  device.

This clip is also the one that caught the most serious bug found across
all three review rounds: a candidate side with zero confident-landmark
settling samples (the walker/demonstrator occludes the body for the whole
~1.2s settling window) once caused a floor-detection guard to trust an
uncalibrated fallback value as if it were real, permanently locking out
every later genuinely-low reading and dropping this clip's rep count to 0
with 220 of ~1843 frames wrongly rejected. Fixed by gating that guard on a
`baseline_calibrated` flag that's only true once settling actually
calibrated real samples — a candidate with no settling data now correctly
falls back to unguarded (but self-correcting) behavior instead of a
permanently broken one.

## Seated knee extension

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
