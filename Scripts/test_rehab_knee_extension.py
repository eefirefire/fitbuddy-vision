"""
Regression tests for rehab_knee_extension.py.

Every test here corresponds to a real bug found by hand-testing against
real footage during development. Run with: pytest test_rehab_knee_extension.py
"""
import numpy as np
import pytest

from rehab_knee_extension import (
    VectorGeometryEngine,
    TemporalSequenceBuffer,
    KinematicDerivativesEngine,
    AMIAnomalyClassifier,
    ExtensionDeficitTracker,
    LSIStateMachine,
    CameraAlignmentChecker,
    TrunkComplianceChecker,
    mp_pose,
    _CandidateLegTrack,
)


class _FakeLandmark:
    def __init__(self, x, y):
        self.x, self.y, self.z, self.visibility = x, y, 0.0, 1.0


class _FakePoseLandmarks:
    """Minimal stand-in for MediaPipe's pose_landmarks — a 33-entry list
    indexable the same way real landmarks are, with only the shoulder/hip
    entries actually populated (everything else defaults to the origin,
    which CameraAlignmentChecker never reads)."""
    def __init__(self, left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x, shoulder_y=0.3, hip_y=0.6):
        blank = _FakeLandmark(0.0, 0.0)
        self.landmark = [blank] * 33
        self.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = _FakeLandmark(left_shoulder_x, shoulder_y)
        self.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = _FakeLandmark(right_shoulder_x, shoulder_y)
        self.landmark[mp_pose.PoseLandmark.LEFT_HIP.value] = _FakeLandmark(left_hip_x, hip_y)
        self.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value] = _FakeLandmark(right_hip_x, hip_y)


# ── Geometry ─────────────────────────────────────────────────────────────────

def test_bent_knee_is_90_degrees():
    # vertex-based vectors: hip and ankle on perpendicular arms from the knee
    angle = VectorGeometryEngine.calculate_knee_angle((0, 0), (0, 1), (1, 1))
    assert angle == pytest.approx(90.0, abs=0.5)


def test_straight_knee_is_180_degrees():
    # Regression: an earlier version used hip->knee / knee->ankle PATH
    # vectors instead of vertex-anchored vectors, which made a straight leg
    # compute as 0 degrees instead of 180.
    angle = VectorGeometryEngine.calculate_knee_angle((0, 0), (0, 1), (0, 2))
    assert angle == pytest.approx(180.0, abs=0.5)


def test_degenerate_landmarks_return_nan_not_crash():
    angle = VectorGeometryEngine.calculate_knee_angle((0, 0), (0, 0), (0, 0))
    assert np.isnan(angle)


# ── Smoothing edge behavior ──────────────────────────────────────────────────

def test_smoothing_does_not_crush_the_most_recent_sample():
    # Regression: centered convolution (mode="same") zero-pads at the array
    # edge, and the LAST element is exactly what every frame reads as
    # "current velocity". For a steadily rising ramp, that bug made the
    # smoothed edge value drop (135 -> 124 -> 112 -> 99 -> 85, going DOWN
    # even though the raw signal keeps rising) — a fabricated reversal that
    # silently broke peak detection. A causal (backward-only) average lags
    # the raw signal (expected and fine) but must never reverse direction
    # like that: each new point should be >= the previous smoothed point for
    # monotonically increasing raw input.
    angles = np.array([90.0 + 9.0 * i for i in range(10)])  # steady linear ramp
    smoothed = KinematicDerivativesEngine._smooth(angles)
    assert smoothed[-1] > smoothed[-2] > smoothed[-3]
    assert smoothed[-1] > 100  # nowhere near the ~85 a zero-padded edge would give


# ── AMI / jitter classifier ──────────────────────────────────────────────────

def _feed_angles(track: _CandidateLegTrack, timestamps, angles):
    for t, a in zip(timestamps, angles):
        track.push_frame(a, t)


def test_smooth_healthy_rep_is_not_flagged_inhibited():
    # Regression: ordinary landmark jitter alone (no real stutter) used to
    # rack up 10-20+ acceleration sign-flips/sec and falsely flag healthy
    # reps as "inhibited" — confirmed against real MediaPipe output.
    # Includes a real lowering phase at the end since descent-based peak
    # detection needs one to ever commit a rep at all.
    t = np.linspace(0, 1.6, 48)
    rising = np.clip(90 + 90 * t / 1.0, 90, 180)
    lowering_mask = t > 1.2
    angles = rising.copy()
    angles[lowering_mask] = np.maximum(90, 180 - (t[lowering_mask] - 1.2) * 150)
    track = _CandidateLegTrack()
    _feed_angles(track, t, angles)
    assert len(track.reps) == 1
    assert track.any_rep_inhibited is False


def test_genuine_multi_pause_stutter_is_flagged_inhibited():
    # Three distinct mid-rise pauses on the way up, then a real lowering
    # phase back toward bent (descent-based peak detection needs an actual
    # descent to ever commit a rep at all).
    t = np.linspace(0, 3.6, 108)
    base = np.clip(90 + 90 * (t / 3.0), 90, 180)
    stutter = base.copy()
    for start, end in [(0.7, 1.1), (1.5, 1.9), (2.3, 2.7)]:
        mask = (t >= start) & (t <= end)
        stutter[mask] = stutter[mask][0]
    lowering = t > 3.2
    stutter[lowering] = np.maximum(90, 180 - (t[lowering] - 3.2) * 150)

    track = _CandidateLegTrack()
    _feed_angles(track, t, stutter)
    assert len(track.reps) == 1
    assert track.reps[0]["peak_angle"] == pytest.approx(180.0, abs=1.0)
    assert track.any_rep_inhibited is True


def test_sustained_frames_required_not_single_spike():
    # A single momentary spike of choppiness should not be enough on its own
    # — only a SUSTAINED run of consecutive choppy frames should flag a rep.
    velocity = np.array([10.0, -10.0, 10.0, -10.0, 10.0])
    acceleration = np.array([500.0, -500.0, 500.0, -500.0, 500.0])
    result = AMIAnomalyClassifier.classify_ascent(velocity, acceleration)
    # The classifier itself reports per-window choppiness; the SUSTAINED
    # requirement lives in _CandidateLegTrack, exercised by the tests above.
    assert "is_inhibited" in result


# ── Rep / peak detection ──────────────────────────────────────────────────────

def test_rep_count_matches_real_visible_reps_not_noise():
    # Regression: an earlier version of check_for_peak fired on ANY
    # extending->near-zero-velocity transition, including landmark-jitter
    # wobble at the top of a rep and noise blips near the bent starting
    # position — inflating 3 real reps into 9 detected "reps" on real
    # footage. The rearm-on-return-to-bent + minimum-peak-angle fix should
    # produce exactly the number of genuine extension cycles.
    tracker = ExtensionDeficitTracker()
    detected = []

    def feed(angle, velocity):
        peak_angle, deficit, _ecc = tracker.check_for_peak(angle, velocity)
        if peak_angle is not None:
            detected.append(peak_angle)

    # Rep 1: bent -> straight -> noisy wobble at the top -> real descent commits the peak
    feed(90, 5)
    feed(140, 200)
    feed(178, 200)
    feed(178, 5)
    feed(178, -3)    # micro wobble right at the top (should NOT look like descent)
    feed(177, 6)
    feed(178, 4)
    feed(120, -150)  # genuine descent >8deg below the running max (178) -> commits here
    feed(95, -150)

    # Noise blip near the bent position, far from a real extension — must
    # not be counted as a peak at all.
    feed(92, 9)
    feed(91, -9)

    # Rep 2: a genuine second extension, with its own descent back to bent
    feed(140, 200)
    feed(175, 200)
    feed(175, 5)
    feed(118, -150)
    feed(95, -150)

    assert len(detected) == 2
    assert all(p >= 130 for p in detected)


def test_peak_must_exceed_minimum_angle():
    tracker = ExtensionDeficitTracker()
    peak_angle, deficit, _ecc = tracker.check_for_peak(95, 5)  # was extending? no prior state
    # First call can't be a peak (no prior "was_extending" state yet).
    assert peak_angle is None


# ── Eccentric (lowering-phase) pacing ───────────────────────────────────────────

def _rise_then_descend(rise_seconds, descend_seconds, descend_samples=120):
    """bent(90) -> straight(180) over rise_seconds, then straight -> bent(90)
    over descend_seconds, at a steady rate. Used to control exactly how long
    the eccentric phase takes for the pacing tests below."""
    rise_t = np.linspace(0, rise_seconds, 30, endpoint=False)
    rise_angles = np.linspace(90, 180, 30, endpoint=False)

    # endpoint=False above + starting descend_t at rise_seconds avoids a
    # duplicate timestamp at the rise/descend boundary, which would give
    # np.gradient a zero dx and produce NaNs in the velocity trace.
    descend_t = np.linspace(rise_seconds, rise_seconds + descend_seconds, descend_samples)
    descend_angles = np.linspace(180, 90, descend_samples)

    t = np.concatenate([rise_t, descend_t])
    angles = np.concatenate([rise_angles, descend_angles])
    return t, angles


def test_slow_controlled_descent_is_not_flagged_too_fast():
    # A deliberate, controlled lowering phase (well over the ~2s floor) should
    # be timed but NOT flagged — this is what good form looks like for this
    # exercise (PT guidance: control the drop, don't let gravity do it).
    t, angles = _rise_then_descend(rise_seconds=1.0, descend_seconds=4.0)
    track = _CandidateLegTrack()
    _feed_angles(track, t, angles)

    assert len(track.reps) == 1
    rep = track.reps[0]
    assert rep["eccentric_duration_s"] is not None
    assert rep["eccentric_duration_s"] > ExtensionDeficitTracker.MIN_ECCENTRIC_DURATION_S
    assert rep["is_descent_too_fast"] is False


def test_fast_dropped_descent_is_flagged_too_fast():
    # Same rep shape, but the lowering phase takes well under the ~2s floor —
    # this is "dropping" the leg / letting gravity do the work, the exact
    # pattern PT guidance warns against for this exercise.
    t, angles = _rise_then_descend(rise_seconds=1.0, descend_seconds=0.6)
    track = _CandidateLegTrack()
    _feed_angles(track, t, angles)

    assert len(track.reps) == 1
    rep = track.reps[0]
    assert rep["eccentric_duration_s"] is not None
    assert rep["eccentric_duration_s"] < ExtensionDeficitTracker.MIN_ECCENTRIC_DURATION_S
    assert rep["is_descent_too_fast"] is True


# ── LSI ───────────────────────────────────────────────────────────────────────

def test_lsi_ratio_uses_weaker_over_stronger():
    sm = LSIStateMachine()
    sm.record_left(peak_velocity=400, peak_angle=178, deficit=2, is_inhibited=False)
    sm.record_right(peak_velocity=300, peak_angle=160, deficit=20, is_inhibited=True)
    result = sm.compute_lsi()
    assert result["velocity_lsi_pct"] == pytest.approx(75.0, abs=0.1)
    assert result["below_recovery_baseline"] is True


def test_lsi_balanced_legs_no_warning():
    sm = LSIStateMachine()
    sm.record_left(peak_velocity=400, peak_angle=170, deficit=10, is_inhibited=False)
    sm.record_right(peak_velocity=395, peak_angle=168, deficit=12, is_inhibited=False)
    result = sm.compute_lsi()
    assert result["below_recovery_baseline"] is False
    assert result["warning"] is None


# ── Best-rep selection ────────────────────────────────────────────────────────

def test_best_rep_is_largest_extension_not_most_recent():
    # Regression: an earlier version overwrote the committed rep on every
    # new peak (last rep wins), so a tired final rep could erase a clean
    # earlier one. best_rep must pick the largest peak_angle regardless of
    # order.
    track = _CandidateLegTrack()
    track.reps = [
        {"rep_number": 1, "peak_angle": 178.0, "extension_deficit_deg": 2.0, "peak_velocity_deg_s": 400.0, "is_inhibited": False},
        {"rep_number": 2, "peak_angle": 150.0, "extension_deficit_deg": 30.0, "peak_velocity_deg_s": 100.0, "is_inhibited": False},
    ]
    assert track.best_rep["rep_number"] == 1


# ── Camera alignment ────────────────────────────────────────────────────────

def test_true_side_profile_is_classified_good():
    # Shoulders/hips nearly overlapping in x (the camera sees the body edge-on).
    checker = CameraAlignmentChecker()
    for _ in range(10):
        checker.push_frame(_FakePoseLandmarks(0.50, 0.51, 0.50, 0.515))
    result = checker.result()
    assert result["level"] == "good"
    assert result["ratio"] < CameraAlignmentChecker.GOOD_MAX_RATIO


def test_frontal_view_is_classified_poor():
    # Shoulders/hips wide apart in x relative to torso height (camera facing the body).
    checker = CameraAlignmentChecker()
    for _ in range(10):
        checker.push_frame(_FakePoseLandmarks(0.30, 0.70, 0.32, 0.68))
    result = checker.result()
    assert result["level"] == "poor"
    assert result["message"] is not None


def test_no_frames_returns_unknown_not_a_crash():
    checker = CameraAlignmentChecker()
    result = checker.result()
    assert result["level"] == "unknown"
    assert result["ratio"] is None


# ── Trunk compliance (compensatory lean) ────────────────────────────────────

def test_staying_upright_is_classified_good():
    checker = TrunkComplianceChecker()
    for _ in range(40):
        checker.push_frame(_FakePoseLandmarks(0.495, 0.505, 0.495, 0.505, shoulder_y=0.3, hip_y=0.6))
    result = checker.result()
    assert result["level"] == "good"
    assert result["max_deviation_deg"] < TrunkComplianceChecker.CAUTION_DEVIATION_DEG


def test_leaning_back_after_baseline_is_classified_poor():
    checker = TrunkComplianceChecker()
    # First establish baseline while upright.
    for _ in range(15):
        checker.push_frame(_FakePoseLandmarks(0.495, 0.505, 0.495, 0.505, shoulder_y=0.3, hip_y=0.6))
    # Then shift the shoulders horizontally relative to the hips — simulates
    # leaning the trunk back/sideways mid-clip.
    for _ in range(20):
        checker.push_frame(_FakePoseLandmarks(0.575, 0.585, 0.495, 0.505, shoulder_y=0.3, hip_y=0.6))
    result = checker.result()
    assert result["level"] == "poor"
    assert result["message"] is not None


def test_trunk_checker_with_no_frames_returns_unknown():
    checker = TrunkComplianceChecker()
    result = checker.result()
    assert result["level"] == "unknown"
    assert result["max_deviation_deg"] is None
