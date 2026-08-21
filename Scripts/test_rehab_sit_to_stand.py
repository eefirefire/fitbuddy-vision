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
