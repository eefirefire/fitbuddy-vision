"""
Regression tests for rehab_sit_to_stand.py.

Mirrors the structure/rigor of test_rehab_knee_extension.py. Every test here
either (a) confirms a piece of logic ported from the already-hardened
knee-extension pipeline still behaves the same way once adapted to a
composite two-joint angle, or (b) checks something genuinely new to
sit-to-stand (the composite-minimum requirement, the bilateral no-LSI state
machine, both-joint-angle recording per rep).

Run with: pytest test_rehab_sit_to_stand.py
"""
import time

import numpy as np
import pytest

from rehab_sit_to_stand import (
    VectorGeometryEngine,
    KinematicDerivativesEngine,
    AMIAnomalyClassifier,
    StandTransitionTracker,
    CameraAlignmentChecker,
    CueChannel,
    SitToStandState,
    RehabSitToStandSession,
    mp_pose,
    _CandidateBodySideTrack,
)


class _FakeLandmark:
    def __init__(self, x, y):
        self.x, self.y, self.z, self.visibility = x, y, 0.0, 1.0


class _FakePoseLandmarks:
    def __init__(self, left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x, shoulder_y=0.3, hip_y=0.6):
        blank = _FakeLandmark(0.0, 0.0)
        self.landmark = [blank] * 33
        self.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = _FakeLandmark(left_shoulder_x, shoulder_y)
        self.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = _FakeLandmark(right_shoulder_x, shoulder_y)
        self.landmark[mp_pose.PoseLandmark.LEFT_HIP.value] = _FakeLandmark(left_hip_x, hip_y)
        self.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value] = _FakeLandmark(right_hip_x, hip_y)


def _ramp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


# ── Geometry: generalized to any three-point vertex, not just the knee ────────

def test_joint_angle_at_90_degrees():
    angle = VectorGeometryEngine.calculate_joint_angle((0, 0), (0, 1), (1, 1))
    assert angle == pytest.approx(90.0, abs=0.5)


def test_joint_angle_straight_is_180():
    angle = VectorGeometryEngine.calculate_joint_angle((0, 0), (0, 1), (0, 2))
    assert angle == pytest.approx(180.0, abs=0.5)


def test_joint_angle_degenerate_returns_nan_not_crash():
    angle = VectorGeometryEngine.calculate_joint_angle((0, 0), (0, 0), (0, 0))
    assert np.isnan(angle)


def test_same_formula_works_at_hip_and_knee_vertex():
    # The whole point of generalizing calculate_knee_angle into
    # calculate_joint_angle was to reuse it at BOTH the hip vertex
    # (shoulder-hip-knee) and the knee vertex (hip-knee-ankle) without two
    # separate implementations. Confirm both actually work with the same
    # function, not just that the knee-shaped call still works.
    hip_angle = VectorGeometryEngine.calculate_joint_angle((0, 0), (0, 1), (1, 2))  # shoulder, hip, knee
    knee_angle = VectorGeometryEngine.calculate_joint_angle((0, 1), (1, 2), (1, 3))  # hip, knee, ankle
    assert 0 <= hip_angle <= 180
    assert 0 <= knee_angle <= 180


# ── Composite-angle requirement: a rep needs BOTH joints extended ─────────────

def test_rep_requires_both_hip_and_knee_extended_not_just_one():
    # Regression guard for the core clinical distinction this module exists
    # to make: pushing up through locked knees without extending the hip
    # (a common compensation) must NOT count as a full stand. The composite
    # standing_angle = min(hip_angle, knee_angle), so if hip stays bent
    # (~100 deg) while knee fully extends (~175 deg), the composite should
    # reflect the WEAKER joint, not the stronger one.
    track = _CandidateBodySideTrack()
    t = 0.0
    # Knee extends fully but hip barely moves -- knee locks out, torso stays
    # folded forward. This should NOT register as a real completed stand.
    hip_seq = _ramp(95, 105, 15) + _ramp(105, 95, 15)   # hip barely moves
    knee_seq = _ramp(95, 175, 15) + _ramp(175, 95, 15)  # knee fully extends
    for hip_a, knee_a in zip(hip_seq, knee_seq):
        track.push_frame(hip_a, knee_a, t)
        t += 1 / 30
    # The composite angle is capped by the hip (the weaker joint), so the
    # peak-acceptance floor (observed_min + margin) should never be cleared
    # by knee motion alone -- no rep should be counted.
    assert len(track.reps) == 0, f"a knee-only compensation should not count as a stand: {track.reps}"


def test_rep_counts_when_both_joints_genuinely_extend():
    track = _CandidateBodySideTrack()
    t = 0.0
    hip_seq = _ramp(95, 175, 15) + _ramp(175, 95, 60)
    knee_seq = _ramp(95, 178, 15) + _ramp(178, 95, 60)
    for hip_a, knee_a in zip(hip_seq, knee_seq):
        track.push_frame(hip_a, knee_a, t)
        t += 1 / 30
    assert len(track.reps) == 1
    rep = track.reps[0]
    # Composite standing_angle should be close to the LOWER of the two
    # joints at peak (hip, since it's the more limiting one here at 175 vs
    # knee's 178).
    assert rep["standing_angle"] == pytest.approx(175.0, abs=1.0)
    # Both individual joint angles must be recorded, not just the composite
    # -- this is what a goniometer comparison needs (see the module's own
    # build-scope note: "tracking hip and knee angle at the top/bottom of
    # the movement").
    assert rep["hip_angle_deg"] is not None
    assert rep["knee_angle_deg"] is not None


# ── Outlier resistance -- same median-filter protection ported from the
#    knee-extension module's ExtensionDeficitTracker ────────────────────────

def test_single_frame_outlier_does_not_corrupt_the_floor():
    tracker = StandTransitionTracker()
    sequence = [95, 96, 94, 95, 30, 95, 96, 94, 95, 96]  # 30 is a one-frame glitch
    t = 0.0
    for a in sequence:
        tracker.check_for_peak(a, t)
        t += 1 / 30
    assert tracker._observed_min_angle > 50, (
        f"a single-frame outlier corrupted the rep floor: {tracker._observed_min_angle}"
    )


def test_implausible_low_reading_is_rejected_outright():
    # Direct unit test of the rejection guard itself
    # (_CandidateBodySideTrack.MAX_DRIFT_BELOW_SETTLED_BASELINE_DEG /
    # push_frame), isolated from the AMI classifier -- see the test below
    # for the end-to-end version against a simulated real glitch.
    track = _CandidateBodySideTrack()
    track.transition_tracker.observe_baseline(95.0)  # settling calibrates a real seated baseline
    track.finalize_settling()  # commits the calibrated baseline (median of settling samples)
    track.push_frame(150.0, 150.0, 0.0)  # first real-tracking frame establishes observed_min
    floor_before = track.transition_tracker._observed_min_angle

    # A wildly implausible low reading (more than MAX_DRIFT_BELOW_SETTLED_
    # BASELINE_DEG below the current floor) must be rejected outright, not
    # merely clamped -- it should never reach observed_min at all. Uses a
    # large dt so the (separate) velocity guard doesn't also fire here --
    # this test targets the absolute-floor guard specifically.
    result = track.push_frame(1.0, 1.0, 5.0)
    assert result.get("rejected_implausible_low_angle") is True
    assert track.transition_tracker._observed_min_angle == floor_before, (
        "an implausible low reading still altered observed_min despite being rejected"
    )


def test_sustained_real_change_is_still_accepted():
    tracker = StandTransitionTracker()
    sequence = [95, 95, 95] + [80, 80, 80, 80, 80]
    t = 0.0
    for a in sequence:
        tracker.check_for_peak(a, t)
        t += 1 / 30
    assert tracker._observed_min_angle <= 81


# ── Deficit clamping -- same fix ported from the knee-extension module ──────

def test_standing_deficit_clamped_to_nonnegative():
    tracker = StandTransitionTracker()
    seq = _ramp(95, 175, 15) + _ramp(175, 95, 15)  # peak (175) exceeds target (165)
    t = 0.0
    last_deficit = None
    for a in seq:
        peak, deficit, ecc = tracker.check_for_peak(a, t)
        if deficit is not None:
            last_deficit = deficit
        t += 1 / 30
    assert last_deficit is not None
    assert last_deficit >= 0, f"deficit went negative: {last_deficit}"


# ── Cue channel -- rapid same-window emissions must both be recoverable ─────

def test_cue_pending_recovers_both_cues_fired_in_one_window():
    ch = CueChannel()
    ch.emit("Stand up further")
    ch.emit("Slow down")
    pending = ch.pending_since(0)
    assert pending == [{"seq": 1, "text": "Stand up further"}, {"seq": 2, "text": "Slow down"}]


def test_cue_pending_empty_once_caught_up():
    ch = CueChannel()
    ch.emit("Good")
    assert ch.pending_since(ch.seq) == []


# ── Bilateral state machine: no left/right split, no LSI ────────────────────

def test_state_machine_has_no_left_right_split():
    # Sit-to-stand is bilateral -- there should be exactly SELECT, RECORD,
    # ANALYZE, none of the per-limb SELECT_LEFT/RECORD_LEFT/PROMPT_SWITCH/
    # SELECT_RIGHT/RECORD_RIGHT states the knee-extension module has.
    states = {s.name for s in SitToStandState}
    assert states == {"SELECT", "RECORD", "ANALYZE"}


def test_advance_state_walks_select_record_analyze():
    session = RehabSitToStandSession()
    assert session.state == SitToStandState.SELECT
    session.advance_state()
    assert session.state == SitToStandState.RECORD
    session.advance_state()
    assert session.state == SitToStandState.ANALYZE
    # No further states beyond ANALYZE.
    session.advance_state()
    assert session.state == SitToStandState.ANALYZE


# ── Settling-period baseline calibration -- ported fix from knee extension ──

def test_settling_calibration_prevents_unreachable_floor():
    session = RehabSitToStandSession()
    session.advance_state()  # -> RECORD
    track = session._candidates["left"]

    # Simulate the settling window: person sitting still at their real
    # baseline (hip/knee both ~95).
    for _ in range(5):
        track.observe_settling_angle(min(95, 96))
    track.finalize_settling()  # commits the calibrated baseline

    # Settling ends with the person ALREADY mid-rise (95 -> 178), not at
    # their seated baseline -- the exact scenario that, without settling
    # calibration, seeds the floor too high and can make peak detection
    # nearly unreachable.
    hip_seq = _ramp(150, 178, 10) + _ramp(178, 95, 30)
    knee_seq = _ramp(150, 178, 10) + _ramp(178, 95, 30)
    t = 0.0
    for hip_a, knee_a in zip(hip_seq, knee_seq):
        track.push_frame(hip_a, knee_a, t)
        t += 1 / 30

    assert track.transition_tracker._observed_min_angle < 100, (
        "settling calibration should have seeded a realistic baseline, not ~150"
    )
    assert len(track.reps) >= 1, "a real rep should still be detected after settling calibration"


# ── Camera alignment -- unchanged reuse, still exercise-agnostic ────────────

def test_true_side_profile_is_classified_good():
    checker = CameraAlignmentChecker()
    for _ in range(5):
        checker.push_frame(_FakePoseLandmarks(0.49, 0.51, 0.49, 0.51))
    assert checker.result()["level"] == "good"


def test_frontal_view_is_classified_poor():
    checker = CameraAlignmentChecker()
    for _ in range(5):
        checker.push_frame(_FakePoseLandmarks(0.2, 0.8, 0.2, 0.8))
    assert checker.result()["level"] == "poor"


# ── AMI/jerk classifier reused generically on the composite signal ──────────

def test_smooth_rise_is_not_flagged_jerky():
    velocity = np.array([10.0, 20.0, 30.0, 25.0, 15.0, 5.0])
    acceleration = np.gradient(velocity)
    result = AMIAnomalyClassifier.classify_motion(velocity, acceleration)
    assert result["is_jerky"] is False


def test_genuine_stutter_is_flagged_jerky():
    # Multiple real direction reversals in the acceleration trace, well
    # past ZERO_ACCELERATION_EPS, repeated -- a real stutter, not noise.
    acceleration = np.array([500, -500, 500, -500, 500, -500, 500, -500])
    velocity = np.cumsum(acceleration) * 0.01
    result = AMIAnomalyClassifier.classify_motion(velocity, acceleration)
    assert result["is_jerky"] is True


# ── Regression: a real still/resting pause must not fire "Slow down" ────────
# Found by running rehab_sit_to_stand against a real ~4.5min video with long
# (13-97s) narrated pauses between reps (sit_to_stand_controlled.mp4) --
# AMIAnomalyClassifier is shape-only (acceleration sign flips), with no
# amplitude floor, so sub-pixel MediaPipe landmark jitter on a person
# standing still talking was enough to register as "jerky" and fire a false
# "Slow down" cue for someone who wasn't moving at all.

def test_resting_jitter_does_not_fire_slow_down():
    track = _CandidateBodySideTrack()
    t = 0.0
    # Small (<1 deg) alternating jitter around a fixed standing angle, no
    # real displacement -- simulates landmark noise while genuinely at rest.
    jitter = [95.0, 95.4, 94.7, 95.3, 94.8, 95.2, 94.9, 95.1] * 15
    for a in jitter:
        track.push_frame(a, a, t)
        t += 1 / 30
    assert track.cue.text is None, f"resting jitter incorrectly fired a cue: {track.cue.text}"


# ── Regression: a MediaPipe tracking-glitch spike must not corrupt the
# permanent rep-detection floor ──────────────────────────────────────────
# Found by running against real footage (sit_to_stand_controlled.mp4): a
# multi-frame glitch swung the composite angle 172 -> 1.1 -> 162 degrees in
# ~0.2s (implied velocities of 1000-3800+ deg/s), while MediaPipe reported
# normal-to-high landmark visibility throughout -- a confidence-score gate
# does not catch this. Because StandTransitionTracker._observed_min_angle
# is a running minimum that never resets, one such spike would otherwise
# permanently corrupt rep detection for the rest of the session: every
# later sway/gesture during a real standing hold would clear the
# now-artificially-low peak/rearm margins and register as a spurious rep.

def test_implausible_glitch_spike_does_not_corrupt_observed_min():
    track = _CandidateBodySideTrack()
    t = 0.0
    # Simulate the settling/calibration window a real session runs before
    # rep tracking starts -- this is what establishes the anchor
    # _CandidateBodySideTrack.MAX_DRIFT_BELOW_SETTLED_BASELINE_DEG uses.
    for _ in range(5):
        track.transition_tracker.observe_baseline(95.0)
    # Genuine rise to a real stand (95 -> 172), then hold near standing --
    # matches the real footage pattern this regression is modeling.
    for a in _ramp(95, 172, 15):
        track.push_frame(a, a, t)
        t += 1 / 30
    for a in [172, 171, 173, 172]:
        track.push_frame(a, a, t)
        t += 1 / 30
    floor_before_glitch = track.transition_tracker._observed_min_angle

    # The glitch: a physically-impossible plunge to ~1 degree and back,
    # all within a few 1/30s frames.
    for a in [82.7, 38.7, 16.7, 1.8, 1.1, 3.9, 8.2, 122.8, 152.7]:
        track.push_frame(a, a, t)
        t += 1 / 30

    # The floor is allowed to drift down by up to MAX_DRIFT_BELOW_SETTLED_
    # BASELINE_DEG (that's the whole point of a bounded, not zero, margin --
    # real per-person variation exists) but must NOT collapse anywhere near
    # the glitch's actual ~1 degree values.
    floor_after_glitch = track.transition_tracker._observed_min_angle
    max_allowed_drop = _CandidateBodySideTrack.MAX_DRIFT_BELOW_SETTLED_BASELINE_DEG
    assert floor_after_glitch >= floor_before_glitch - max_allowed_drop, (
        "a tracking-glitch spike corrupted the permanent rep-detection floor beyond "
        f"the allowed drift: {floor_before_glitch} -> {floor_after_glitch}"
    )

    # Confirm the cascading symptom is also gone: a normal small sway while
    # still standing (well above any real seated baseline) must NOT
    # register as a brand new rep once the glitch has passed.
    reps_before_sway = len(track.reps)
    for a in [172, 160, 175, 165, 173]:
        track.push_frame(a, a, t)
        t += 1 / 30
    assert len(track.reps) == reps_before_sway, (
        "ordinary standing sway after a glitch was miscounted as a new rep"
    )


def test_genuine_slow_stutter_still_fires_slow_down():
    # Same "three distinct mid-rise pauses" construction used in
    # test_rehab_knee_extension.test_genuine_multi_pause_stutter_is_flagged_inhibited
    # -- a real stop-start rise (multi-tenths-of-a-second holds, not
    # frame-to-frame alternation) must still be caught after the motion
    # floor is added, not just resting-still jitter.
    t = np.linspace(0, 3.6, 108)
    base = np.clip(95 + 85 * (t / 3.0), 95, 180)
    stutter = base.copy()
    for start, end in [(0.7, 1.1), (1.5, 1.9), (2.3, 2.7)]:
        mask = (t >= start) & (t <= end)
        stutter[mask] = stutter[mask][0]

    track = _CandidateBodySideTrack()
    for ts, a in zip(t, stutter):
        track.push_frame(a, a, ts)
    assert track.cue.text == "Slow down", f"a genuine stutter should still fire Slow down, got: {track.cue.text}"
