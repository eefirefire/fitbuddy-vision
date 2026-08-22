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


class _FakeFullPoseLandmarks:
    """A pose with real left/right shoulder/hip/knee/ankle landmarks, for
    exercising RehabSitToStandSession.route_frame/process_frame end-to-end
    (unlike _FakePoseLandmarks above, which only has shoulder/hip -- enough
    for CameraAlignmentChecker but not for joint-angle extraction).

    Uses two fixed poses (a shoulder offset that bends the hip vs. a
    straight vertical line) rather than deriving exact landmark positions
    from a target angle via trigonometry -- these tests only need
    real-vs-fake landmark data to exercise the settling/routing plumbing,
    not precise angle values, so exact degrees don't matter here."""

    _BENT = {"shoulder": (0.55, 0.20), "hip": (0.50, 0.50), "knee": (0.50, 0.75), "ankle": (0.50, 0.95)}
    _STRAIGHT = {"shoulder": (0.50, 0.20), "hip": (0.50, 0.50), "knee": (0.50, 0.75), "ankle": (0.50, 0.95)}

    def __init__(self, pose: str = "straight", side: str = "right"):
        blank = _FakeLandmark(0.0, 0.0)
        self.landmark = [blank] * 33
        points = self._BENT if pose == "bent" else self._STRAIGHT
        sides = ("left", "right") if side == "both" else (side,)
        for s in sides:
            prefix = "LEFT" if s == "left" else "RIGHT"
            for joint in ("shoulder", "hip", "knee", "ankle"):
                idx = getattr(mp_pose.PoseLandmark, f"{prefix}_{joint.upper()}").value
                self.landmark[idx] = _FakeLandmark(*points[joint])


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
    # finalize_settling() must actually run (not just observe_baseline) --
    # a previous version of this test omitted that call, so its settling
    # simulation was silently inert and _observed_min_angle was actually
    # established by the ramp's own running-min instead, by coincidence
    # landing on the same value and masking the gap.
    for _ in range(5):
        track.transition_tracker.observe_baseline(95.0)
    track.finalize_settling()
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


# ── Regression: settling calibration must prime the median-of-3 outlier
# filter, not just observed_min_angle ────────────────────────────────────
# Found via code review: observe_baseline used to feed _update_observed_min
# directly (pre-populating _recent_angles with real settling samples by the
# time active tracking began). After switching to a settling-window median
# (finalize_baseline), _recent_angles started genuinely empty at the
# settling/active boundary, so the first 1-2 real check_for_peak calls
# computed a median over just 1-2 values -- effectively no outlier
# protection right when tracking starts, despite real settling data
# existing to seed it with.

def test_finalize_baseline_primes_median_filter_for_immediate_protection():
    tracker = StandTransitionTracker()
    for angle in [95.0, 96.0, 94.0]:
        tracker.observe_baseline(angle)
    tracker.finalize_baseline()
    assert len(tracker._recent_angles) == 3, (
        "finalize_baseline should seed _recent_angles from the settling "
        f"samples, got {list(tracker._recent_angles)}"
    )

    # With the filter primed, a single wild outlier on the very FIRST real
    # tracking frame must not corrupt the floor -- this is the actual
    # end-to-end symptom the priming exists to prevent.
    tracker.check_for_peak(30.0, 0.0)  # a one-frame glitch, right at the boundary
    assert tracker._observed_min_angle > 50, (
        "a single-frame outlier on the first post-settling frame corrupted "
        f"the floor: {tracker._observed_min_angle}"
    )


# ── Regression: the settling window's contamination protection is the
# whole-window MEDIAN, not a per-sample rejection guard ─────────────────────
# A per-sample "deviation from recent settling samples" guard was tried and
# reverted: if the very first settling sample is itself the contamination,
# every later genuine sample gets judged against that one bad reference and
# rejected too, with no way to recover -- confirmed via code review to make
# calibration WORSE than doing nothing in that ordering. The median in
# finalize_baseline is what actually protects against a MINORITY of bad
# samples; these tests confirm that property directly, not a per-sample
# check that no longer exists.

def test_settling_median_tolerates_a_single_contaminated_sample():
    tracker = StandTransitionTracker()
    for angle in [95.0, 96.0, 94.0, 95.0]:
        tracker.observe_baseline(angle)
    # A wild single-sample glitch during settling itself.
    tracker.observe_baseline(2.0)
    tracker.observe_baseline(95.0)
    tracker.finalize_baseline()
    assert tracker._observed_min_angle > 50, (
        "a single implausible settling-window sample corrupted the calibrated "
        f"baseline despite being a clear minority: {tracker._observed_min_angle}"
    )


def test_settling_median_even_tolerates_a_contaminated_first_sample():
    # The specific case a per-sample guard got wrong: contamination as the
    # VERY FIRST reading. The whole-window median handles this fine as
    # long as it's still a minority of the total settling samples.
    tracker = StandTransitionTracker()
    tracker.observe_baseline(18.0)  # the documented real-footage contamination value
    for angle in [95.0, 96.0, 94.0, 95.0, 93.0]:
        tracker.observe_baseline(angle)
    tracker.finalize_baseline()
    assert tracker._observed_min_angle > 50, (
        f"a contaminated FIRST settling sample corrupted the calibrated baseline: {tracker._observed_min_angle}"
    )


def test_settling_accepts_genuine_gradual_movement():
    # No per-sample guard exists to over-reject a person genuinely still
    # adjusting position during settling -- every sample should reach
    # finalize_baseline's median.
    tracker = StandTransitionTracker()
    for angle in [110.0, 105.0, 100.0, 96.0, 93.0]:
        tracker.observe_baseline(angle)
    tracker.finalize_baseline()
    assert tracker._observed_min_angle == pytest.approx(100.0), (
        f"genuine gradual settling movement produced an unexpected baseline: {tracker._observed_min_angle}"
    )


# ── Regression: a glitch confined to ONE joint must not slip past the
# composite-only rejection guards ────────────────────────────────────────
# Found via code review: both push_frame guards check only the composite
# standing_angle = min(hip, knee). A glitch on the joint that ISN'T
# currently the composite minimum is invisible to them, and the raw
# glitched value can be captured verbatim into a rep's clinical output.

def test_glitch_confined_to_non_limiting_joint_is_rejected():
    track = _CandidateBodySideTrack()
    t = 0.0
    # Genuine rise where hip is the limiting (smaller) joint throughout.
    for hip_a, knee_a in zip(_ramp(95, 150, 15), _ramp(95, 178, 15)):
        track.push_frame(hip_a, knee_a, t)
        t += 1 / 30
    # A single-frame glitch confined to knee_angle -- hip_angle (still the
    # composite minimum) is completely normal, so a composite-only guard
    # would see nothing wrong.
    result = track.push_frame(151.0, 400.0, t)
    assert result.get("rejected_implausible_jump") is True, (
        f"a knee-only implausible jump was not rejected: {result}"
    )
    # And it must never have reached the peak-tracking state.
    assert track._knee_at_running_peak is None or track._knee_at_running_peak < 200, (
        f"glitched knee_angle leaked into _knee_at_running_peak: {track._knee_at_running_peak}"
    )


# ── Regression: a non-monotonic timestamp must be rejected, not silently
# skip the velocity guard entirely ───────────────────────────────────────

def test_non_monotonic_timestamp_is_rejected_not_silently_passed():
    track = _CandidateBodySideTrack()
    track.push_frame(150.0, 150.0, 5.0)
    # A duplicate timestamp (dt == 0) with a wildly different angle used to
    # completely bypass the velocity guard (the `if dt > 0` check simply
    # skipped, letting the frame through unchecked).
    result = track.push_frame(10.0, 10.0, 5.0)
    assert result.get("rejected_non_monotonic_timestamp") is True, (
        f"a duplicate/non-monotonic timestamp was not rejected: {result}"
    )
    assert track.rejected_frame_count == 1


def test_out_of_order_timestamp_is_also_rejected():
    track = _CandidateBodySideTrack()
    track.push_frame(150.0, 150.0, 5.0)
    result = track.push_frame(10.0, 10.0, 4.0)  # earlier than the last accepted frame
    assert result.get("rejected_non_monotonic_timestamp") is True


# ── Documents a known scoping limit: the velocity guard is only an
# effective defense against a near-single-frame teleport ────────────────
# Found via code review: _last_accepted_timestamp only advances on accept,
# so dt against it grows during a run of rejections, and implied_speed =
# |delta| / dt shrinks for the same angle change as dt grows. Investigated
# capping the effective dt to stop this, but the fix doesn't actually work:
# given MAX_PLAUSIBLE_VELOCITY_DEG_S=2500 and the anatomical maximum
# possible delta (180 degrees), no within-range value can ever exceed this
# guard once dt is more than ~0.07s regardless of any cap -- the velocity
# guard is, by construction, already scoped to near-single-frame gaps
# only. Sustained/gradual drift (staleness included) is caught instead by
# MAX_DRIFT_BELOW_SETTLED_BASELINE_DEG, which is time-independent. This
# test documents that scoping rather than asserting a fix that can't work.

def test_velocity_guard_only_catches_near_single_frame_jumps():
    track = _CandidateBodySideTrack()
    track.push_frame(150.0, 150.0, 0.0)
    # A genuine single-frame-interval gap: the full anatomical range in
    # ~1/30s is caught.
    result = track.push_frame(-30.0, -30.0, 1 / 30)
    assert result.get("rejected_implausible_jump") is True

    track2 = _CandidateBodySideTrack()
    track2.push_frame(150.0, 150.0, 0.0)
    # The same magnitude of change over a real multi-second gap is
    # legitimately plausible (a real slow movement could cover this range
    # in that time) and is correctly NOT treated as a velocity violation --
    # it's the absolute-floor guard's job to catch an implausible RESULT,
    # not the velocity guard's job to catch every large change over a long
    # span.
    result2 = track2.push_frame(-30.0, -30.0, 10.0)
    assert result2.get("rejected_implausible_jump") is not True


# ── Regression: rejected frames must be counted somewhere, not silently
# dropped with zero observability ────────────────────────────────────────

def test_rejected_frames_are_counted_and_surfaced_in_final_report():
    session = RehabSitToStandSession()
    session.advance_state()  # -> RECORD
    track = session._candidates["right"]
    # Give "right" enough real range of motion that _decide_tracked_
    # landmark_side picks it over "left" (untouched, 0 ROM) -- otherwise
    # finalize_recording would report on the wrong (unused) candidate.
    for a in _ramp(95, 172, 15):
        track.push_frame(a, a, len(track._angle_samples) / 30)
    track.push_frame(400.0, 400.0, track._last_accepted_timestamp + 0.03)  # implausible jump, rejected
    session.advance_state()  # -> ANALYZE (finalizes recording)
    report = session.final_report()
    assert "rejected_frame_count" in report
    assert report["rejected_frame_count"] >= 1


# ── Regression: settling must be a single choke point ALL frame sources
# go through, not reimplemented (or omitted) per caller ──────────────────
# Found via code review: the /upload Flask route fed frames straight into
# process_frame from frame 0 with no settling step at all, reintroducing
# the exact uncalibrated-floor bug the settling mechanism exists to fix.
# route_frame is now the one method every frame source must call.

def test_route_frame_settles_before_tracking_starts():
    session = RehabSitToStandSession()
    session.advance_state()  # -> RECORD
    settling_window_s = 0.5

    # First frame establishes the settling window's start; frames within
    # the window must not reach process_frame/push_frame at all.
    contaminated = _FakeFullPoseLandmarks(pose="bent", side="both")
    for i in range(5):
        result = session.route_frame(contaminated, 640, 480, i * 0.05, settling_window_s)
        assert result == {}, "a frame during the settling window should produce no tracking result"

    track = session._candidates["right"]
    assert track.transition_tracker._observed_min_angle == float("inf"), (
        "settling frames should accumulate into _baseline_samples, not observed_min_angle, until finalize"
    )

    # A frame past the settling window should finalize calibration and
    # start real tracking.
    real_frame = _FakeFullPoseLandmarks(pose="straight", side="both")
    session.route_frame(real_frame, 640, 480, settling_window_s + 0.01, settling_window_s)
    assert track.transition_tracker._observed_min_angle != float("inf"), (
        "the first post-settling frame should have triggered finalize_settling"
    )


def test_route_frame_is_the_only_settling_implementation_upload_and_batch_share():
    # Direct proof that upload_video and run_batch_validation both route
    # through the same method (rather than each hand-rolling their own
    # settling branch) -- inspects the source rather than re-running a full
    # video, since that's expensive; this is a structural regression guard.
    import inspect
    import rehab_sit_to_stand as sts
    upload_src = inspect.getsource(sts.upload_video)
    batch_src = inspect.getsource(sts.run_batch_validation)
    assert "route_frame" in upload_src, "upload_video must route through route_frame, not a bespoke settling branch"
    assert "route_frame" in batch_src, "run_batch_validation must route through route_frame, not a bespoke settling branch"


# ── Regression: live_status's settling flag must not diverge from what
# route_frame/finalize_settling actually did ─────────────────────────────
# Found via code review: live_status used to recompute its OWN independent
# wall-clock settling formula instead of reading route_frame's state, so it
# could report settling:False (and expose uncalibrated defaults) before
# finalize_settling had actually run, if pose detection dropped out right
# at the grace-period boundary.

def test_is_settling_reflects_route_frame_state_not_a_separate_clock():
    session = RehabSitToStandSession()
    session.advance_state()  # -> RECORD
    # Before any frame has been routed, is_settling must not claim settling
    # ended just because wall-clock time has passed -- there is no
    # calibration to have "ended".
    assert session.is_settling(time.time() + 100) is True

    frame = _FakeFullPoseLandmarks(pose="bent", side="both")
    session.route_frame(frame, 640, 480, 0.0, 0.5)
    assert session.is_settling(0.1) is True  # still within the 0.5s window
    assert session._settling_finalized is False  # not yet -- window hasn't ended

    # A frame whose timestamp is past the window actually ends settling and
    # triggers finalize_settling as a side effect of route_frame itself.
    session.route_frame(frame, 640, 480, 0.6, 0.5)
    assert session.is_settling(0.6) is False
    assert session._settling_finalized is True


# ── Regression: the exact bug this pass's own fix was written to prevent
# (baseline_calibrated staying False for a whole session) must have direct
# coverage, not just coverage of the calibrated-baseline case ─────────────
# Found via a third code-review pass: every existing settling test always
# seeds real samples, so baseline_calibrated was never exercised as False --
# a regression back to the prior bug (gating on `observed_min != inf`
# instead) would pass every other test in this file.

def test_drift_guard_stays_inert_when_settling_produces_zero_samples():
    track = _CandidateBodySideTrack()
    # Settling produces NOTHING for this side (e.g. landmarks never
    # cleared MIN_LANDMARK_VISIBILITY throughout the whole window --
    # confirmed reachable against real footage: sit_to_stand_walker.mp4).
    track.finalize_settling()
    assert track.transition_tracker.baseline_calibrated is False

    # A frame reaches active tracking anyway (e.g. via the old fallback
    # path once check_for_peak runs) and happens to seed a HIGH value.
    for a in [170.0, 171.0, 172.0]:
        track.push_frame(a, a, len(track._angle_samples) / 30)

    # A genuinely low (e.g. real seated) reading afterward must NOT be
    # rejected just because it's far below that uncalibrated value -- the
    # drift guard has nothing real to protect yet.
    result = track.push_frame(60.0, 60.0, 1.0)
    assert result.get("rejected_implausible_low_angle") is not True, (
        f"the drift guard fired despite baseline_calibrated being False: {result}"
    )


def test_drift_guard_activates_once_settling_actually_calibrates():
    track = _CandidateBodySideTrack()
    track.transition_tracker.observe_baseline(95.0)
    track.finalize_settling()
    assert track.transition_tracker.baseline_calibrated is True

    track.push_frame(150.0, 150.0, 0.0)
    result = track.push_frame(1.0, 1.0, 5.0)
    assert result.get("rejected_implausible_low_angle") is True


# ── Regression: a session must not let two frame sources with different
# timestamp bases corrupt each other's settling decision ────────────────
# Found via code review: the live stream uses wall-clock time.time() and
# /upload uses video-relative seconds starting near 0. Nothing previously
# stopped both from feeding the same shared global session (e.g. a user
# starts a live stream, then also POSTs an upload, before advancing state)
# -- whichever reached route_frame first would anchor _route_started_at in
# its own time basis, and the other source's settling check would then be
# computed against a wildly mismatched scale.

def test_route_frame_ignores_a_second_source_once_one_has_claimed_the_session():
    session = RehabSitToStandSession()
    session.advance_state()  # -> RECORD
    frame = _FakeFullPoseLandmarks(pose="straight", side="both")

    # "live" claims the session first, using wall-clock-scale timestamps.
    session.route_frame(frame, 640, 480, 1_700_000_000.0, 3.0, source="live")
    assert session._frame_source == "live"

    # "upload" tries to feed a frame with a wildly different (video-
    # relative) timestamp scale -- must be ignored outright, not allowed to
    # reach settling/active tracking and corrupt _route_started_at.
    track = session._candidates["right"]
    samples_before = len(track.transition_tracker._baseline_samples)
    result = session.route_frame(frame, 640, 480, 0.03, 0.5, source="upload")
    assert result == {}
    assert len(track.transition_tracker._baseline_samples) == samples_before, (
        "a mismatched-source frame was still processed instead of ignored"
    )
    assert session._route_started_at == 1_700_000_000.0, (
        "a mismatched-source frame corrupted the session's established timestamp basis"
    )


def test_route_frame_allows_repeated_calls_from_the_same_source():
    session = RehabSitToStandSession()
    session.advance_state()  # -> RECORD
    frame = _FakeFullPoseLandmarks(pose="straight", side="both")
    session.route_frame(frame, 640, 480, 0.0, 0.5, source="upload")
    # A second frame from the SAME source must be processed normally.
    result = session.route_frame(frame, 640, 480, 0.6, 0.5, source="upload")
    assert session._frame_source == "upload"
    # Not asserting on result contents here (depends on tracking specifics) --
    # only that it wasn't silently dropped by the source guard.
    assert session._settling_finalized is True


def test_route_frame_source_resets_on_a_new_recording():
    session = RehabSitToStandSession()
    session.advance_state()  # -> RECORD
    frame = _FakeFullPoseLandmarks(pose="straight", side="both")
    session.route_frame(frame, 640, 480, 0.0, 0.5, source="live")
    assert session._frame_source == "live"

    session.advance_state()  # -> ANALYZE
    session.advance_state()  # no-op past ANALYZE, state machine caps here
    # A brand new session (the real-world equivalent of /reset) starts fresh.
    fresh = RehabSitToStandSession()
    assert fresh._frame_source is None
