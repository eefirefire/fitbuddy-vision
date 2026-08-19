# Video Tests

Real, unscripted video clips used to regression-test the live rehab pipeline
(`Scripts/rehab_knee_extension.py`) end-to-end — real MediaPipe pose
detection, real human movement, run through the actual live-camera code path
instead of synthetic angle arrays. Built while debugging the Tier 1 live
rep-counter/voice-cue feature; kept here so the same scenarios can be
re-checked after future changes without re-hunting for source clips.

**Important caveat:** none of these are real seated knee-extension footage —
this app has no such clips available, and there's no camera here to record
one. These are squat videos, used purely as stand-ins for "a real person
moving in front of a camera" to exercise the mechanics (rep detection, cue
triggering, state transitions). Two consequences:
- `camera_alignment` will correctly read `"poor"` on all of these, since
  none are a proper side-on seated shot — that's expected, not a bug.
- Peak-angle/deficit numbers exercise the code paths correctly (the angle
  math is exercise-agnostic) but aren't clinically meaningful readings.

Before trusting any real clinical validation, retest against actual seated
knee-extension footage of a real patient/volunteer.

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
