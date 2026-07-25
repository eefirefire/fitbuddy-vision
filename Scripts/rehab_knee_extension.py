"""
FitBuddy-AI — Seated Knee Extension Rehab Module
=================================================

Clinical context
-----------------
The Seated Knee Extension (SKE) is a closed-chain isolation exercise used
ubiquitously in post-op ACL/meniscus/TKA rehab protocols. The patient sits
sideways to the camera, knee bent ~90 deg, and extends the shin outward to
~180 deg (a straight leg). Because the patient sits *side-on* to the camera,
the hip, knee and ankle all move within a single 2D sagittal plane for the
entire repetition — there is no limb-crossing, no foreshortening, and no
depth ambiguity the way there is in e.g. a frontal-view squat. This is why
this exercise is treated as "tracking-glitch resistant": MediaPipe's 2D
landmark estimate for HIP/KNEE/ANKLE degrades mostly from occlusion or
foreshortening, and a perfect side profile structurally avoids both.

This module is not a rep counter. It is a kinematic diagnostic pipeline that:
  1. Extracts the 2D knee joint angle every frame (planar vector geometry).
  2. Maintains a rolling ~1.5s temporal buffer of angle/time samples.
  3. Differentiates that buffer to get angular velocity and acceleration.
  4. Classifies the acceleration profile as healthy vs. neuromotor-inhibited
     (Arthrogenic Muscle Inhibition / pain-guarding) using clinically
     informed velocity and "choppiness" thresholds.
  5. Detects the rep's peak extension angle and reports the Extension
     Deficit relative to the 180 deg clinical baseline.
  6. Runs a two-sided (L/R) state machine and computes a Limb Symmetry
     Index (LSI) once both legs have been recorded.

The module exposes a Flask app for live/streamed use, but every stage is a
plain, independently testable class so unit tests / notebooks can drive the
math without touching Flask or OpenCV at all.
"""

import os
import time
import enum
import uuid
import tempfile
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, jsonify, request

from rehab_auth import auth_bp, init_db, login_required

# ─────────────────────────────────────────────────────────────────────────────
# 1. PLANAR ANGULAR GEOMETRY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class VectorGeometryEngine:
    """
    Pure vector math — no MediaPipe, no OpenCV, no I/O.

    The knee angle is the angle *at the knee vertex* between the thigh
    (hip->knee) and shin (knee->ankle) segments. Using the dot-product
    formulation (rather than atan2-difference) keeps the result naturally
    bounded to [0, 180] without the sign-wrap correction atan2-based
    approaches need, which matters here because we feed this straight into
    a derivative pipeline — any angle-wrap discontinuity would show up as a
    fake velocity spike.
    """

    @staticmethod
    def calculate_knee_angle(hip_xy, knee_xy, ankle_xy) -> float:
        """
        theta = arccos( (v1 . v2) / (|v1| |v2|) )

        v1 = knee -> hip   (thigh vector, pointing away from the vertex)
        v2 = knee -> ankle (shin vector, pointing away from the vertex)

        Both vectors originate AT the knee vertex (not chained hip->knee->ankle
        path vectors) — this is what makes arccos return the true interior
        joint angle. A bent knee has these two vectors near-perpendicular
        (~90 deg); a straight leg has them pointing in opposite directions
        (~180 deg), since hip and ankle end up on opposite sides of the knee.
        """
        hip = np.asarray(hip_xy, dtype=np.float64)
        knee = np.asarray(knee_xy, dtype=np.float64)
        ankle = np.asarray(ankle_xy, dtype=np.float64)

        thigh_vec = hip - knee
        shin_vec = ankle - knee

        thigh_norm = np.linalg.norm(thigh_vec)
        shin_norm = np.linalg.norm(shin_vec)
        if thigh_norm == 0 or shin_norm == 0:
            # Degenerate frame (landmarks collapsed to a point) — caller should
            # discard this sample rather than feed a divide-by-zero into the buffer.
            return float("nan")

        cosine = np.dot(thigh_vec, shin_vec) / (thigh_norm * shin_norm)
        cosine = np.clip(cosine, -1.0, 1.0)  # guard against fp drift past arccos domain
        angle_rad = np.arccos(cosine)

        # The dot-product formula returns the *interior* angle between the two
        # segments. For this exercise that is exactly the clinical knee angle
        # (90 deg bent -> 180 deg straight), so no 180-minus correction needed.
        return float(np.degrees(angle_rad))


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEMPORAL SEQUENCE WINDOW BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class TemporalSequenceBuffer:
    """
    Rolling window of (timestamp, angle) samples. maxlen=45 at a nominal
    30 FPS capture rate gives ~1.5s of history — enough to see a full
    ascent-or-descent phase of a slow-tempo rehab rep without holding
    unbounded memory for a long session.
    """

    def __init__(self, maxlen: int = 45):
        self.maxlen = maxlen
        self._timestamps = deque(maxlen=maxlen)
        self._angles = deque(maxlen=maxlen)

    def push(self, timestamp: float, angle: float) -> None:
        if np.isnan(angle):
            return  # drop degenerate samples, never let NaN poison the buffer
        self._timestamps.append(timestamp)
        self._angles.append(angle)

    def as_arrays(self):
        """Returns (timestamps, angles) as numpy arrays, oldest-first."""
        return np.array(self._timestamps), np.array(self._angles)

    def is_ready(self, min_samples: int = 3) -> bool:
        return len(self._angles) >= min_samples

    def __len__(self):
        return len(self._angles)


# ─────────────────────────────────────────────────────────────────────────────
# 3. KINEMATIC DERIVATIVES ENGINE (CALCULUS MODELING)
# ─────────────────────────────────────────────────────────────────────────────
class KinematicDerivativesEngine:
    """
    Differentiates the temporal buffer to angular velocity (deg/s) and
    angular acceleration (deg/s^2). np.gradient is used instead of a naive
    diff() because it is centered (second-order accurate) at interior points,
    which reduces derivative noise from per-frame landmark jitter compared
    to backward-differencing.

    That alone is not enough, though: testing this against real MediaPipe
    output (not synthetic clean curves) showed raw per-frame landmark jitter
    — a pixel or two of sub-frame noise in the knee keypoint — survives the
    centered-difference velocity step and then gets roughly SQUARED in
    effect once you differentiate a second time for acceleration. On real
    footage this alone produced 10-20+ acceleration sign flips per second
    even during objectively smooth, healthy reps — enough to blow through
    the AMI "choppiness" threshold on every single clip, healthy or not.
    The fix is to smooth the angle signal itself before differentiating: a
    short moving average removes the high-frequency jitter component while
    barely touching the low-frequency arc of the actual rep (a real
    extension takes a third of a second or more; landmark jitter flickers
    frame to frame), so genuine multi-pause "stutter" patterns still survive
    smoothing while single-frame noise does not.
    """

    # Width of the moving-average smoothing window applied to the raw angle
    # signal before differentiating. Tuned against real MediaPipe output
    # (not synthetic clean curves): 5 frames still let single/double-frame
    # jitter through; 9 frames (~0.3s at 30fps) suppresses it while still
    # preserving a real pause/stutter, which clinically lasts several
    # tenths of a second — much longer than one frame.
    SMOOTHING_WINDOW = 9

    @classmethod
    def _smooth(cls, values: np.ndarray) -> np.ndarray:
        """
        Causal (backward-looking) moving average: smoothed[i] is the mean of
        values[i] and up to SMOOTHING_WINDOW-1 samples BEFORE it. Never looks
        ahead, because there is no "ahead" in a live/real-time feed — we only
        ever know the past and the current instant.

        This matters more than it might look: an earlier version used a
        centered convolution (np.convolve(..., mode="same")), which zero-pads
        at the array edges. That meant the LAST element of the smoothed
        array — which is exactly the "current" value this pipeline reads
        every frame via velocity[-1] — was averaged against phantom future
        zeros and got crushed toward 0. That silently broke peak detection
        (a real velocity of 470 deg/s would smooth down to a tiny fraction of
        that right at the edge we actually read), which is why this needed
        to be a trailing average instead of a centered one.
        """
        window = min(cls.SMOOTHING_WINDOW, len(values))
        if window < 2:
            return values
        cumsum = np.cumsum(values, dtype=float)
        smoothed = np.empty_like(values, dtype=float)
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            total = cumsum[i] - (cumsum[lo - 1] if lo > 0 else 0.0)
            smoothed[i] = total / (i - lo + 1)
        return smoothed

    @classmethod
    def compute_velocity(cls, timestamps: np.ndarray, angles: np.ndarray) -> np.ndarray:
        if len(angles) < 2:
            return np.zeros_like(angles)
        smoothed_angles = cls._smooth(angles)
        return np.gradient(smoothed_angles, timestamps)  # deg / s

    @classmethod
    def compute_acceleration(cls, timestamps: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        if len(velocity) < 2:
            return np.zeros_like(velocity)
        smoothed_velocity = cls._smooth(velocity)
        return np.gradient(smoothed_velocity, timestamps)  # deg / s^2


# ─────────────────────────────────────────────────────────────────────────────
# 4. AI JITTER ANOMALY CLASSIFIER (AMI / PAIN-COMPENSATION DETECTION)
# ─────────────────────────────────────────────────────────────────────────────
class AMIAnomalyClassifier:
    """
    Classifies a single ascent (extension) phase as a healthy motor pattern
    or as showing signs of Arthrogenic Muscle Inhibition (AMI) / pain
    compensation.

    Clinical reasoning — and a correction from an earlier version of this
    classifier:
      An earlier version of this class flagged a rep as "inhibited" if its
      peak angular velocity fell below ~340 deg/s, on the assumption that a
      healthy quad always fires in one fast, explosive burst (360-470 deg/s).
      That assumption does NOT hold for seated knee extension specifically:
      real physical-therapy guidance for this exact exercise instructs
      patients to move SLOWLY and under control ("avoid swinging the leg,"
      "don't kick it up fast") — so a slow, well-controlled rep and a
      slow, pain-guarded rep can both sit well under 340 deg/s, and an
      absolute speed floor cannot tell them apart. Using it as a pass/fail
      signal would falsely flag correct, careful technique as pathological.

      What DOES still distinguish AMI/pain-guarding from a clean controlled
      rep, at any speed, is the SHAPE of the acceleration trace. A healthy
      rep — fast or slow — accelerates smoothly into the motion and
      decelerates smoothly into the peak: at most one sign change in
      acceleration. An AMI/pain-guarded rep "stutters": the nervous system
      intermittently inhibits the contraction (reflex inhibition from joint
      effusion or pain receptors), producing repeated micro-pauses that show
      up as the acceleration trace repeatedly flipping sign within a single
      ascent. That shape-based signal is speed-independent, so it is the
      only criterion this classifier uses for is_inhibited. Peak velocity is
      still reported (it's useful clinical context — e.g. tracking whether a
      patient's pace changes over a recovery program) but it no longer
      drives the warning on its own.

      A second correction, also found by testing against real footage
      (not just synthetic curves): the "zero" epsilon and flip threshold
      below were originally guessed, and on real MediaPipe data they were
      far too tight. Measuring actual acceleration magnitudes across a real
      clip showed values ranging from near 0 (idle) to several thousand
      deg/s^2 (mid-rep) with an interquartile range of roughly 10-275 — so
      an epsilon of 5 treated almost ALL of that range as "real" sign
      changes, and a real healthy rep racked up 10-20+ flips per second
      purely from jitter. ZERO_ACCELERATION_EPS=100 and a per-window
      threshold of 3 were chosen by sweeping both this real clip and a
      synthetic rep with genuine multi-tenths-of-a-second pauses injected,
      measuring the longest run of CONSECUTIVE over-threshold frames each
      produced: the real clip never sustained more than 0 consecutive
      flagged frames at these settings, while the synthetic stutter
      sustained 6 — clean separation, paired with SUSTAINED_FRAMES_REQUIRED
      below as the actual decision boundary.
    """

    # Number of accel sign-flips within one rolling buffer window beyond
    # which a single frame's evidence counts as "choppy". This is NOT, by
    # itself, enough to flag a whole rep — see SUSTAINED_FRAMES_REQUIRED in
    # _CandidateLegTrack: a real stutter persists across many consecutive
    # frames, so requiring sustained evidence (rather than flagging the
    # instant any one window crosses this) is what actually separates a
    # genuine pain-guarded pattern from a rare noise spike.
    OSCILLATION_FLIP_THRESHOLD = 3

    # Below this acceleration magnitude (deg/s^2) we treat the signal as
    # "roughly zero" rather than a real direction change — see class
    # docstring for how this was derived from real data, not guessed.
    ZERO_ACCELERATION_EPS = 100.0

    @classmethod
    def _count_acceleration_sign_flips(cls, acceleration: np.ndarray) -> int:
        """
        Counts how many times the acceleration trace crosses zero. We ignore
        flips smaller than ZERO_ACCELERATION_EPS so noise around zero doesn't
        get over-counted as a "flip".
        """
        eps = cls.ZERO_ACCELERATION_EPS
        signs = np.sign(np.where(np.abs(acceleration) < eps, 0.0, acceleration))
        signs = signs[signs != 0]  # drop the "roughly zero" stretches entirely
        if len(signs) < 2:
            return 0
        return int(np.sum(np.diff(signs) != 0))

    @classmethod
    def classify_ascent(cls, velocity: np.ndarray, acceleration: np.ndarray) -> dict:
        if len(velocity) == 0:
            return {
                "peak_velocity_deg_s": 0.0,
                "oscillation_flips": 0,
                "is_inhibited": False,
                "warning": None,
            }

        peak_velocity = float(np.max(velocity))
        flips = cls._count_acceleration_sign_flips(acceleration)

        # Speed-independent: only the choppiness of the acceleration trace
        # drives this flag now (see class docstring for why peak velocity
        # alone was removed as a criterion).
        is_inhibited = flips >= cls.OSCILLATION_FLIP_THRESHOLD

        warning = (
            "Unstable Neuromotor Activation (AMI / Pain Compensation Detected)"
            if is_inhibited else None
        )

        return {
            "peak_velocity_deg_s": round(peak_velocity, 1),
            "oscillation_flips": flips,
            "is_inhibited": is_inhibited,
            "warning": warning,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. AUTOMATED EXTENSION DEFICIT CAPTURE
# ─────────────────────────────────────────────────────────────────────────────
class ExtensionDeficitTracker:
    """
    Tracks the running highest angle reached since the leg last left the
    bent starting position, and commits that as the rep's peak once the
    angle genuinely DESCENDS away from it (the eccentric/lowering phase
    beginning) — not merely the moment velocity first touches ~0.

    Why descent, not a velocity-near-zero instant: an earlier version
    treated ANY transition from "extending" to "velocity near zero" as the
    peak. That works for a single clean rep, but fails on a realistic
    AMI/pain-guarded rep that pauses mid-rise and then keeps climbing toward
    a higher true peak — the mid-rise pause got committed as "the peak"
    immediately (it's a real near-zero-velocity instant too), the tracker
    then disarmed until the leg returned to the bent position, and the
    actual higher peak later in the same rep was never seen — silently
    truncating the rep and losing exactly the AMI evidence this whole
    pipeline exists to catch. Requiring a real, sustained drop below the
    running max — not just a momentary stall — means a mid-rise stutter
    doesn't end the rep; the tracker keeps watching for a possibly-higher
    peak until the leg actually starts lowering back down.
    """

    IDEAL_FULL_EXTENSION_DEG = 180.0

    # The leg must visibly return to (near) the bent starting position before
    # the detector "re-arms" to look for the next rep's peak. Without this,
    # testing against real footage showed two failure modes: (1) ordinary
    # landmark jitter right at the TOP of one real rep can make the angle
    # wobble, with each wobble mis-counted as a separate "rep"; (2) a
    # momentary noise dip during the early, low-angle part of a genuine
    # extension (knee still near 90 deg, nowhere near a real peak) could
    # itself look like a tiny "extend then stop" cycle and get counted as a
    # phantom rep. Requiring a real return toward the bent position before
    # allowing the next detection makes both impossible.
    #
    # REARM was originally a fixed absolute angle (110 deg), tuned against
    # one adult demo clip whose resting seated angle was ~90-95 deg. Real
    # multi-subject validation footage broke this: one child's resting bent
    # angle sat around 118-125 deg (different chair height/camera framing),
    # which never dropped below the fixed 110 deg threshold — so only the
    # FIRST of six real reps in that clip was ever detected; the tracker
    # stayed permanently disarmed afterward. Rearming relative to each
    # person's own OBSERVED minimum angle (rather than an assumed absolute
    # number) generalizes across different chair heights, camera distances,
    # and body proportions instead of silently failing whenever someone's
    # baseline doesn't match the one clip this was originally tuned on.
    REARM_MARGIN_DEG = 15.0

    # A genuine rep peak only ever means something past the midpoint of the
    # 90->180 deg range — testing against real footage produced a phantom
    # "peak" reported at 89 degrees (not extended at all) before this floor
    # was added. Anything below this is never eligible to be a peak.
    MIN_PEAK_ANGLE_DEG = 130.0

    # How far the angle must drop below its running max for this rep before
    # we treat that max as final and commit it. Small enough to catch a real
    # lowering phase promptly, large enough that ordinary landmark jitter
    # (a degree or two of noise) at the top of a rep can't look like descent.
    DESCENT_MARGIN_DEG = 8.0

    # Minimum acceptable duration for the eccentric (lowering) phase — from
    # the moment a rep's peak extension is committed to the moment the leg
    # returns to the bent starting position. Like BALLISTIC_VELOCITY_CEILING
    # in rehabInterpret.js, this is a derived proxy, not a clinically
    # validated cutoff for this specific exercise: PT guidance for seated
    # knee extension consistently says to control the drop rather than let
    # gravity do the work, and general resistance-training tempo guidance
    # (ACSM, ~1-2s per phase for novice/intermediate lifters) puts a
    # controlled eccentric in roughly this range. Anything faster is flagged
    # as "dropped, not lowered" — worth saying out loud if asked.
    MIN_ECCENTRIC_DURATION_S = 2.0

    def __init__(self):
        self._armed = True
        self._running_max_angle = float("-inf")
        self._observed_min_angle = float("inf")
        # Timestamp the most recent peak was committed at, so the NEXT
        # rearm (leg back at the bent position) can compute how long that
        # rep's descent took. None when there's no pending descent to time.
        self._peak_timestamp = None

    def check_for_peak(self, current_angle: float, current_velocity: float, timestamp: float = None):
        """
        Returns (peak_angle, extension_deficit, eccentric_duration_s).

        peak_angle/extension_deficit are set the instant a new peak is
        committed, otherwise None. eccentric_duration_s is set on the
        (later) frame where the PREVIOUS rep's descent finishes — i.e. the
        leg returns to the bent starting position — otherwise None. The two
        are independent signals that can each fire on different frames, so
        callers should check them separately rather than assuming they
        arrive together.

        Call this once per frame with the latest angle/velocity/timestamp
        sample. current_velocity is accepted for backward-compatible call
        sites but no longer drives the peak decision — see class docstring
        for why angle-based descent is more robust. timestamp is optional
        only for backward compatibility with old call sites that don't time
        the eccentric phase; pass it to get eccentric_duration_s.
        """
        self._observed_min_angle = min(self._observed_min_angle, current_angle)

        eccentric_duration_s = None
        if current_angle <= self._observed_min_angle + self.REARM_MARGIN_DEG:
            if self._peak_timestamp is not None and timestamp is not None:
                # Cast to plain Python float — callers may pass numpy scalar
                # timestamps (e.g. from a numpy timestamp array), and a numpy
                # float/bool downstream isn't JSON-serializable by Flask's
                # default encoder.
                eccentric_duration_s = round(float(timestamp) - float(self._peak_timestamp), 2)
            self._peak_timestamp = None
            self._armed = True
            self._running_max_angle = float("-inf")  # fresh climb for the next rep

        if self._armed:
            self._running_max_angle = max(self._running_max_angle, current_angle)

        descended = (self._running_max_angle - current_angle) >= self.DESCENT_MARGIN_DEG
        peak_detected = (
            self._armed
            and descended
            and self._running_max_angle >= self.MIN_PEAK_ANGLE_DEG
        )

        if not peak_detected:
            return None, None, eccentric_duration_s

        self._armed = False  # stays disarmed until the leg returns to the bent position
        self._peak_timestamp = timestamp
        peak_angle = self._running_max_angle
        deficit = round(self.IDEAL_FULL_EXTENSION_DEG - peak_angle, 1)
        return round(peak_angle, 1), deficit, eccentric_duration_s


# ─────────────────────────────────────────────────────────────────────────────
# 6. DUAL-LIMB SYMMETRY INDEX (LSI) STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────
class RehabState(enum.Enum):
    SELECT_LEFT = "STATE_SELECT_LEFT"
    RECORD_LEFT = "STATE_RECORD_LEFT"
    PROMPT_SWITCH = "STATE_PROMPT_SWITCH"
    SELECT_RIGHT = "STATE_SELECT_RIGHT"
    RECORD_RIGHT = "STATE_RECORD_RIGHT"
    ANALYZE = "STATE_ANALYZE"


# The linear order the state machine walks through. `advance()` simply moves
# one step forward through this tuple; the analysis is finalized once we land
# on ANALYZE and don't move further.
_STATE_ORDER = (
    RehabState.SELECT_LEFT,
    RehabState.RECORD_LEFT,
    RehabState.PROMPT_SWITCH,
    RehabState.SELECT_RIGHT,
    RehabState.RECORD_RIGHT,
    RehabState.ANALYZE,
)


@dataclass
class LimbRecord:
    """Per-leg cache of the clinically relevant peak parameters."""
    peak_velocity_deg_s: float = 0.0
    peak_extension_angle_deg: float = 0.0
    extension_deficit_deg: float = 0.0
    is_inhibited: bool = False
    # Every individual rep detected during this leg's recording, not just
    # the best one — preserved here because RehabSession resets its
    # per-recording trackers once the next leg starts, so this is the only
    # place the full rep-by-rep history survives through to final_report().
    reps: list = field(default_factory=list)


@dataclass
class LSIStateMachine:
    """
    Drives the bilateral SELECT->RECORD->SWITCH->SELECT->RECORD->ANALYZE flow
    and computes the Limb Symmetry Index once both sides are recorded.

    LSI is computed on peak angular velocity — the standard return-to-sport
    metric in knee rehab literature — and reported alongside an angle-based
    LSI as a secondary cross-check, since a patient can be fast but still
    short of full extension.
    """
    state: RehabState = RehabState.SELECT_LEFT
    left: LimbRecord = field(default_factory=LimbRecord)
    right: LimbRecord = field(default_factory=LimbRecord)

    LSI_CLINICAL_RECOVERY_BASELINE = 90.0  # % — standard return-to-sport threshold

    def advance(self) -> RehabState:
        idx = _STATE_ORDER.index(self.state)
        if idx < len(_STATE_ORDER) - 1:
            self.state = _STATE_ORDER[idx + 1]
        return self.state

    def record_left(self, peak_velocity, peak_angle, deficit, is_inhibited, reps=None):
        self.left = LimbRecord(peak_velocity, peak_angle, deficit, is_inhibited, reps or [])

    def record_right(self, peak_velocity, peak_angle, deficit, is_inhibited, reps=None):
        self.right = LimbRecord(peak_velocity, peak_angle, deficit, is_inhibited, reps or [])

    def compute_lsi(self) -> dict:
        """
        LSI = (Peak Param of weaker leg / Peak Param of stronger leg) * 100%
        Computed independently for velocity and peak-angle so a clinician can
        see *which* dimension is driving any asymmetry.
        """
        def ratio(a, b):
            if a == 0 and b == 0:
                return 0.0
            weaker, stronger = sorted([a, b])
            if stronger == 0:
                return 0.0
            return round((weaker / stronger) * 100.0, 1)

        velocity_lsi = ratio(self.left.peak_velocity_deg_s, self.right.peak_velocity_deg_s)
        angle_lsi = ratio(self.left.peak_extension_angle_deg, self.right.peak_extension_angle_deg)

        below_baseline = (
            velocity_lsi < self.LSI_CLINICAL_RECOVERY_BASELINE
            or angle_lsi < self.LSI_CLINICAL_RECOVERY_BASELINE
        )

        return {
            "velocity_lsi_pct": velocity_lsi,
            "angle_lsi_pct": angle_lsi,
            "below_recovery_baseline": below_baseline,
            "warning": (
                f"Limb Symmetry Index below the {self.LSI_CLINICAL_RECOVERY_BASELINE:.0f}% "
                "clinical recovery baseline — asymmetric strength/ROM recovery detected."
                if below_baseline else None
            ),
            "left": self.left,
            "right": self.right,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7. POSE LANDMARK EXTRACTION (MediaPipe)
# ─────────────────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Side-profile landmark indices we need per active leg.
_LANDMARK_SETS = {
    "left": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    "right": (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
}


def extract_hip_knee_ankle(pose_landmarks, side: str, frame_w: int, frame_h: int):
    """
    Pulls the 2D pixel-space (x, y) coordinates for HIP/KNEE/ANKLE of the
    given side. Normalized MediaPipe coords (0-1) are scaled to pixel space
    purely so on-screen overlays line up; the angle math itself is scale
    invariant so this scaling has zero effect on the computed angle.
    """
    hip_lm, knee_lm, ankle_lm = _LANDMARK_SETS[side]
    lm = pose_landmarks.landmark

    def to_xy(landmark_enum):
        p = lm[landmark_enum.value]
        return (p.x * frame_w, p.y * frame_h)

    return to_xy(hip_lm), to_xy(knee_lm), to_xy(ankle_lm)


class CameraAlignmentChecker:
    """
    Detects whether the camera is actually filming a clean side profile
    (sagittal plane), which the whole angle-measurement pipeline assumes.
    Every other fix in this module deals with what happens once a side-on
    clip is correctly tracked — this one catches the upstream case where the
    clip was never side-on to begin with, which silently skews every angle
    downstream with no other symptom.

    The signal: in a true side profile, the left and right shoulders (and
    left and right hips) sit almost directly behind/in front of each other
    from the camera's view, so their horizontal (x) separation in the 2D
    projection is small. As the camera rotates toward a frontal view, that
    separation grows toward the body's real shoulder/hip width. Comparing
    that horizontal separation to the torso's vertical extent (shoulder-to-
    hip height, which stays roughly stable under this kind of rotation)
    gives a single, scale-invariant ratio that tracks how "side-on" the shot
    actually is.

    There is no peer-reviewed source for what counts as "side-on enough" for
    this purpose, so the thresholds below were derived empirically: measured
    directly against a confirmed-clean side-on rehab clip (mean ratio ~0.16,
    p90 ~0.17) versus several frontal/angled exercise clips (ratios 0.33-0.51)
    from unrelated footage. The bands leave a margin on both sides of that
    real gap rather than splitting it down the middle, to avoid flagging a
    genuinely good side-on shot just because of ordinary per-frame noise.
    """

    GOOD_MAX_RATIO = 0.25       # below this: clearly side-on
    POOR_MIN_RATIO = 0.40       # at/above this: clearly not side-on
    # Between the two: "caution" — probably off-axis, but not unusable.

    def __init__(self):
        self._ratios = []

    def push_frame(self, pose_landmarks) -> None:
        lm = pose_landmarks.landmark
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

        torso_height = abs((ls.y + rs.y) / 2 - (lh.y + rh.y) / 2)
        if torso_height < 0.01:
            return  # degenerate frame (landmarks collapsed) — skip rather than divide by ~0

        shoulder_width = abs(ls.x - rs.x)
        hip_width = abs(lh.x - rh.x)
        ratio = (shoulder_width + hip_width) / 2 / torso_height
        self._ratios.append(ratio)

    def result(self) -> dict:
        if not self._ratios:
            return {"ratio": None, "level": "unknown", "message": None}

        mean_ratio = float(np.mean(self._ratios))
        if mean_ratio < self.GOOD_MAX_RATIO:
            level, message = "good", None
        elif mean_ratio < self.POOR_MIN_RATIO:
            level, message = "caution", (
                "Your camera angle looks slightly off from a true side profile. "
                "Angle measurements assume the camera is perpendicular to your body — "
                "try squaring the camera up to your side before your next clip."
            )
        else:
            level, message = "poor", (
                "Your camera doesn't look like it's filming a side profile. "
                "All the angle measurements in this app assume a side-on view, so results "
                "from this clip may be significantly skewed — reposition the camera directly "
                "to your side (perpendicular to your body) and re-record."
            )
        return {"ratio": round(mean_ratio, 3), "level": level, "message": message}


class TrunkComplianceChecker:
    """
    Flags compensatory trunk lean — rocking the torso backward to use
    momentum/body weight instead of the quad to drive the leg up. This is
    one of the most commonly cited form faults for this exact exercise: the
    real "Ask Doctor Jo"-style demo videos we tested against this project
    against explicitly list "Leaning Backward" alongside "Using Momentum"
    and "Forcefully Locking the Knee" as the three mistakes to avoid for
    seated/terminal knee extension. Multiple PT exercise guides repeat the
    same instruction in their own words — see classifyMovementSpeed in
    rehabInterpret.js for the citations already gathered for this exercise
    family. None of them give a specific degree threshold, though, so this
    checker measures deviation from the PATIENT'S OWN seated baseline rather
    than against an absolute "must be perfectly vertical" assumption —
    natural resting posture varies person to person, but a real compensation
    pattern shows up as a clear CHANGE from wherever that person started.

    The deviation thresholds below were derived empirically (not from a
    clinical source — said explicitly here rather than implied): measured
    directly against the same clean, controlled side-on clip already used to
    derive the camera-alignment thresholds. Trunk angle (shoulder-midpoint
    to hip-midpoint vector vs. vertical) stayed within about 2.2 degrees of
    its own mean across an entire 3-rep clip with good form (std ~0.6 deg).
    The bands below add a several-times-larger margin on top of that
    measured noise floor before calling anything a real compensation
    pattern, specifically so ordinary landmark jitter or natural sway never
    triggers a false flag.
    """

    BASELINE_FRAMES = 15      # frames used to establish "this patient's normal seated angle"
    CAUTION_DEVIATION_DEG = 5.0
    POOR_DEVIATION_DEG = 10.0

    def __init__(self):
        self._baseline_samples = []
        self._baseline_angle = None
        self._max_deviation = 0.0

    @staticmethod
    def _trunk_angle_from_vertical(pose_landmarks) -> float:
        lm = pose_landmarks.landmark
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_mid = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
        hip_mid = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])
        trunk_vec = shoulder_mid - hip_mid
        norm = np.linalg.norm(trunk_vec)
        if norm < 1e-6:
            return float("nan")

        vertical = np.array([0.0, -1.0])  # "up" in image coordinates (y grows downward)
        cosine = np.dot(trunk_vec, vertical) / norm
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def push_frame(self, pose_landmarks) -> None:
        angle = self._trunk_angle_from_vertical(pose_landmarks)
        if np.isnan(angle):
            return

        if self._baseline_angle is None:
            self._baseline_samples.append(angle)
            if len(self._baseline_samples) >= self.BASELINE_FRAMES:
                self._baseline_angle = float(np.mean(self._baseline_samples))
            return

        deviation = abs(angle - self._baseline_angle)
        self._max_deviation = max(self._max_deviation, deviation)

    def result(self) -> dict:
        if self._baseline_angle is None:
            return {"max_deviation_deg": None, "level": "unknown", "message": None}

        deviation = round(self._max_deviation, 1)
        if deviation < self.CAUTION_DEVIATION_DEG:
            level, message = "good", None
        elif deviation < self.POOR_DEVIATION_DEG:
            level, message = "caution", (
                f"Your trunk shifted about {deviation}° from your starting seated position during this clip. "
                "A small amount of sway is normal, but try to keep your torso still and let your quad — not your "
                "body weight — do the work."
            )
        else:
            level, message = "poor", (
                f"Your trunk shifted about {deviation}° from your starting seated position during this clip. "
                "This usually means leaning back or rocking the body to help swing the leg up, which several "
                "physical therapy guides for this exact exercise warn against. Try sitting tall and isolating "
                "the movement to the knee."
            )
        return {"max_deviation_deg": deviation, "level": level, "message": message}


class _CandidateLegTrack:
    """
    Independent kinematics pipeline for ONE candidate MediaPipe landmark
    side ('left' or 'right'). RehabSession runs two of these in parallel for
    every frame of a recording — see _resolve_tracked_landmark_side for why
    we can't decide up front which one is the real, camera-facing leg.

    Running both the whole time (instead of spending the first N frames in
    a "calibration window" deciding which to trust) means we never throw
    away the early part of a rep just because the decision wasn't made yet
    — important for short clips where the real motion might BE those early
    frames.
    """

    # A real AMI/pain-guarded stutter is SUSTAINED — it shows up across many
    # consecutive frames, not as a single momentary spike. Flagging the whole
    # rep the instant any one rolling-buffer window crosses the flip
    # threshold means even a very low per-frame false-positive rate becomes
    # near-certain to trigger at least once across hundreds of frames in a
    # clip (this was confirmed by testing: a 0.2%-per-frame false-positive
    # rate still flagged a real, smooth clip simply because it ran long
    # enough). Requiring the threshold to hold for several consecutive
    # frames before committing to "inhibited" is what actually distinguishes
    # a genuine pattern from noise.
    SUSTAINED_FRAMES_REQUIRED = 5

    def __init__(self):
        self.buffer = TemporalSequenceBuffer(maxlen=45)
        self.deficit_tracker = ExtensionDeficitTracker()
        self.min_angle_seen = float("inf")
        self.max_angle_seen = float("-inf")
        # All raw angle samples for this candidate, used only by
        # range_of_motion below (percentile range needs the full
        # distribution, not just the running min/max).
        self._angle_samples: list[float] = []
        # One entry per detected rep (extension peak), in the order they
        # occurred — not just the single best one. A patient's later reps
        # getting slower or shorter as they fatigue is itself a clinically
        # useful signal that a single "best rep" summary throws away.
        self.reps: list[dict] = []
        self._current_rep_peak_velocity = 0.0
        self._current_rep_inhibited = False
        self._consecutive_choppy_frames = 0

    @property
    def range_of_motion(self) -> float:
        """
        Used to decide which candidate side is the real, camera-facing leg
        (see RehabSession._decide_tracked_landmark_side). Originally this was
        raw max-min, which real multi-subject footage broke: a SINGLE
        one-frame landmark-jitter spike on the wrong (occluded) side gave it
        a larger raw range than the correct side's six genuine, repeatable
        extension cycles, causing the wrong leg to be selected. A percentile
        range (90th minus 10th percentile) is far less sensitive to one
        outlier frame — it asks "how much does this side typically move,"
        not "what's the single most extreme value this side ever produced."
        """
        if len(self._angle_samples) < 5:
            if self.max_angle_seen == float("-inf"):
                return 0.0
            return self.max_angle_seen - self.min_angle_seen  # not enough samples yet for a stable percentile
        return float(np.percentile(self._angle_samples, 90) - np.percentile(self._angle_samples, 10))

    @property
    def best_rep(self) -> dict | None:
        """The rep with the largest extension reached — used wherever the
        pipeline previously needed a single representative number (e.g. the
        LSI calculation, which compares one number per leg)."""
        if not self.reps:
            return None
        return max(self.reps, key=lambda r: r["peak_angle"])

    @property
    def any_rep_inhibited(self) -> bool:
        return any(r["is_inhibited"] for r in self.reps)

    def push_frame(self, angle: float, timestamp: float) -> dict:
        self.min_angle_seen = min(self.min_angle_seen, angle)
        self.max_angle_seen = max(self.max_angle_seen, angle)
        self._angle_samples.append(angle)

        self.buffer.push(timestamp, angle)
        if not self.buffer.is_ready(min_samples=3):
            return {"angle": round(angle, 1)}

        timestamps, angles = self.buffer.as_arrays()
        velocity = KinematicDerivativesEngine.compute_velocity(timestamps, angles)
        acceleration = KinematicDerivativesEngine.compute_acceleration(timestamps, velocity)

        ami_result = AMIAnomalyClassifier.classify_ascent(velocity, acceleration)
        current_velocity = float(velocity[-1])

        self._current_rep_peak_velocity = max(self._current_rep_peak_velocity, current_velocity)
        if ami_result["is_inhibited"]:
            self._consecutive_choppy_frames += 1
        else:
            self._consecutive_choppy_frames = 0
        if self._consecutive_choppy_frames >= self.SUSTAINED_FRAMES_REQUIRED:
            self._current_rep_inhibited = True

        peak_angle, deficit, completed_eccentric_s = self.deficit_tracker.check_for_peak(
            angle, current_velocity, timestamp
        )
        if peak_angle is not None:
            # Commit THIS rep's record, then reset the per-rep accumulators
            # so the next rep starts with a clean slate rather than carrying
            # over this one's velocity/inhibition evidence. eccentric_duration_s
            # isn't known yet at this instant (the descent hasn't happened) —
            # it gets backfilled below once the leg actually returns to the
            # bent position.
            self.reps.append({
                "rep_number": len(self.reps) + 1,
                "peak_angle": round(peak_angle, 1),
                "extension_deficit_deg": round(deficit, 1) if deficit is not None else 0.0,
                "peak_velocity_deg_s": round(self._current_rep_peak_velocity, 1),
                "is_inhibited": self._current_rep_inhibited,
                "eccentric_duration_s": None,
                "is_descent_too_fast": False,
            })
            self._current_rep_peak_velocity = 0.0
            self._current_rep_inhibited = False
            self._consecutive_choppy_frames = 0

        if completed_eccentric_s is not None and self.reps:
            # This frame is where the PREVIOUS rep's descent finished, so
            # backfill its timing onto that already-committed rep record.
            self.reps[-1]["eccentric_duration_s"] = completed_eccentric_s
            self.reps[-1]["is_descent_too_fast"] = bool(
                completed_eccentric_s < ExtensionDeficitTracker.MIN_ECCENTRIC_DURATION_S
            )

        return {
            "angle": round(angle, 1),
            "velocity_deg_s": round(current_velocity, 1),
            "ami": ami_result,
            "extension_deficit_deg": deficit,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. SESSION ORCHESTRATOR — ties geometry + buffer + derivatives + classifiers
#    + state machine together for one continuous capture (live cam or file)
# ─────────────────────────────────────────────────────────────────────────────
class RehabSession:
    """
    One full rehab session: walks the LSI state machine, feeding frames from
    either a live webcam or a video file into the kinematics pipeline,
    rendering a debug overlay, and producing a final clinical report.
    """

    def __init__(self):
        self.state_machine = LSIStateMachine()
        self._candidates = {"left": _CandidateLegTrack(), "right": _CandidateLegTrack()}
        self._detected_landmark_side = None  # decided lazily, see _decide_tracked_landmark_side

    def _active_side(self) -> str | None:
        if self.state_machine.state == RehabState.RECORD_LEFT:
            return "left"
        if self.state_machine.state == RehabState.RECORD_RIGHT:
            return "right"
        return None

    def _reset_side_tracking(self):
        self._candidates = {"left": _CandidateLegTrack(), "right": _CandidateLegTrack()}
        self._detected_landmark_side = None

    def advance_state(self) -> RehabState:
        # Commit whatever was just recorded (live-streaming path) BEFORE
        # leaving a RECORD state — the upload/batch paths also call
        # finalize_recording() explicitly right after their frame loop, so
        # this is a no-op for them (committing twice is harmless: it just
        # recomputes the same winning leg and overwrites with identical data).
        if self.state_machine.state in (RehabState.RECORD_LEFT, RehabState.RECORD_RIGHT):
            self.finalize_recording()

        new_state = self.state_machine.advance()
        if new_state in (RehabState.RECORD_LEFT, RehabState.RECORD_RIGHT):
            self._reset_side_tracking()
        return new_state

    def _decide_tracked_landmark_side(self) -> str:
        """
        In a single-camera side-profile shot, the leg facing AWAY from the
        camera is occluded — but MediaPipe still happily emits a confident-
        looking (high visibility) estimate for it by guessing from torso
        pose priors. Visibility alone can't tell us which leg is real, so
        instead we compare the total range of angular motion each candidate
        has shown SO FAR: the leg that isn't moving is the occluded guess,
        the leg that swings through a real arc is the genuine one. Re-run
        every frame (cheap) so the decision keeps improving as more of the
        clip is seen, rather than being locked in by a fixed window.
        """
        left_rom = self._candidates["left"].range_of_motion
        right_rom = self._candidates["right"].range_of_motion
        self._detected_landmark_side = "left" if left_rom >= right_rom else "right"
        return self._detected_landmark_side

    def process_frame(self, pose_landmarks, frame_w: int, frame_h: int, timestamp: float) -> dict:
        """
        Feeds one frame's pose landmarks through BOTH candidate legs' tracking
        pipelines (so no data is ever discarded while we figure out which leg
        is real), and returns the live overlay numbers for whichever side
        currently looks like the genuine, camera-facing leg. No-ops outside
        of the RECORD_LEFT / RECORD_RIGHT states.
        """
        ui_side = self._active_side()
        if ui_side is None or pose_landmarks is None:
            return {}

        frame_results = {}
        for candidate in ("left", "right"):
            hip, knee, ankle = extract_hip_knee_ankle(pose_landmarks, candidate, frame_w, frame_h)
            angle = VectorGeometryEngine.calculate_knee_angle(hip, knee, ankle)
            if np.isnan(angle):
                continue
            frame_results[candidate] = self._candidates[candidate].push_frame(angle, timestamp)

        active_side = self._decide_tracked_landmark_side()
        return frame_results.get(active_side, {})

    def finalize_recording(self):
        """
        Call once a RECORD_LEFT / RECORD_RIGHT clip has been fully processed.
        Commits whichever candidate leg showed real motion into the LSI
        state machine. Must be called explicitly (rather than committing
        per-frame) so a clip that starts with a few ambiguous/noisy frames
        doesn't get locked onto the wrong leg before enough evidence is in.
        """
        ui_side = self._active_side()
        if ui_side is None:
            return
        winner = self._decide_tracked_landmark_side()
        track = self._candidates[winner]
        best = track.best_rep

        record_fn = self.state_machine.record_left if ui_side == "left" else self.state_machine.record_right
        record_fn(
            peak_velocity=best["peak_velocity_deg_s"] if best else 0.0,
            peak_angle=best["peak_angle"] if best else 0.0,
            deficit=best["extension_deficit_deg"] if best else 0.0,
            is_inhibited=track.any_rep_inhibited,
            reps=track.reps,
        )

    def final_report(self) -> dict:
        lsi = self.state_machine.compute_lsi()
        return {
            "state": self.state_machine.state.value,
            "left": lsi["left"].__dict__,
            "right": lsi["right"].__dict__,
            "velocity_lsi_pct": lsi["velocity_lsi_pct"],
            "angle_lsi_pct": lsi["angle_lsi_pct"],
            "below_recovery_baseline": lsi["below_recovery_baseline"],
            "warning": lsi["warning"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 9. VIDEO SOURCE — MOCK / LIVE TOGGLE
# ─────────────────────────────────────────────────────────────────────────────
class VideoSource:
    """
    Wraps cv2.VideoCapture so the rest of the pipeline doesn't care whether
    frames are coming from a webcam or a recorded validation clip.

    Pass `source=0` (or any int) for a live camera, or a filesystem path for
    a recorded video — this is the single switch needed for automated
    regression testing against a fixed library of clips vs. live demoing.
    """

    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# 10. FLASK ROUTING / STATE-MACHINE LOOP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Needed to sign the session cookie used by the login system (rehab_auth.py).
# Dev-only: a real deployment must set this from a secret manager, not a
# literal in source.
app.secret_key = os.environ.get("REHAB_SECRET_KEY", "dev-only-fitbuddy-secret-change-me")
# Login sessions ride on cookies, and browsers refuse to send cookies on a
# cross-origin request unless the response opts in explicitly per-origin —
# a wildcard "*" origin (the previous setting) is REJECTED by browsers for
# credentialed requests, so this must be the exact frontend origin instead.
FRONTEND_ORIGIN = os.environ.get("REHAB_FRONTEND_ORIGIN", "http://localhost:5173")

init_db()
app.register_blueprint(auth_bp)


@app.after_request
def _allow_cors(response):
    # Dev-only: lets the Vite frontend call this Flask backend (port 5050)
    # across origins, with cookies, without a proxy.
    response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# A single in-memory session for demo purposes. A production deployment would
# key this by session/user id (e.g. in Redis) instead of a module-level global.
_session = RehabSession()
_pose_model = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Flip this to an int (e.g. 0) to stream from a live webcam instead of a file.
VIDEO_SOURCE = os.environ.get("REHAB_VIDEO_SOURCE", "0")
try:
    VIDEO_SOURCE = int(VIDEO_SOURCE)
except ValueError:
    pass  # it's a file path, leave as string


def _draw_overlay(frame, frame_result: dict, state: RehabState):
    cv2.putText(frame, f"STATE: {state.value}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if "angle" in frame_result:
        cv2.putText(frame, f"Knee Angle: {frame_result['angle']:.1f} deg", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if "velocity_deg_s" in frame_result:
        cv2.putText(frame, f"Velocity: {frame_result['velocity_deg_s']:.1f} deg/s", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    ami = frame_result.get("ami")
    if ami and ami.get("warning"):
        cv2.putText(frame, ami["warning"], (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return frame


def _mjpeg_generator():
    """Pulls frames from VIDEO_SOURCE, runs the full pipeline, yields MJPEG chunks."""
    source = VideoSource(VIDEO_SOURCE)
    try:
        while True:
            ret, frame = source.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = _pose_model.process(image_rgb)

            frame_result = {}
            if results.pose_landmarks:
                frame_result = _session.process_frame(results.pose_landmarks, w, h, time.time())
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            frame = _draw_overlay(frame, frame_result, _session.state_machine.state)

            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
    finally:
        source.release()


@app.route("/api/rehab/stream")
@login_required
def stream():
    """MJPEG live overlay stream — point an <img> tag at this endpoint."""
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/rehab/upload", methods=["POST"])
@login_required
def upload_video():
    """
    Accepts a recorded clip (multipart field 'video') and runs it through the
    pipeline for whichever leg the state machine currently expects
    (RECORD_LEFT or RECORD_RIGHT). This is the upload-based counterpart to
    the live /stream endpoint — same RehabSession/process_frame path, just
    fed from a saved file instead of a webcam.
    """
    if _session.state_machine.state not in (RehabState.RECORD_LEFT, RehabState.RECORD_RIGHT):
        return jsonify({
            "error": f"Cannot record a video while in state {_session.state_machine.state.value}. "
                     "Click 'Next' to move into a RECORD state first."
        }), 400

    uploaded = request.files.get("video")
    if uploaded is None:
        return jsonify({"error": "No video file in request (expected field 'video')."}), 400

    suffix = os.path.splitext(uploaded.filename or "clip.mp4")[1] or ".mp4"
    tmp_path = os.path.join(tempfile.gettempdir(), f"rehab_{uuid.uuid4().hex}{suffix}")
    uploaded.save(tmp_path)

    frames_processed = 0
    frames_with_pose = 0
    alignment_checker = CameraAlignmentChecker()
    trunk_checker = TrunkComplianceChecker()
    try:
        source = VideoSource(tmp_path)
        # IMPORTANT: an uploaded file is read and pose-estimated as fast as the
        # CPU allows — much faster than real time, and that speed varies run
        # to run. Stamping frames with wall-clock time.time() would therefore
        # measure "how fast this machine processed the file" rather than "how
        # fast the leg actually moved." We instead derive each frame's
        # timestamp from its position in the video's own timeline
        # (frame_index / fps), which is deterministic and matches what a
        # human watching the clip would perceive.
        fps = source.cap.get(cv2.CAP_PROP_FPS) or 30.0
        # IMPORTANT: use a FRESH Pose instance for this upload rather than the
        # shared module-level _pose_model. MediaPipe's tracker deliberately
        # keeps internal temporal state between .process() calls (so a live
        # stream tracks smoothly) — but that means reusing one instance across
        # unrelated uploaded clips leaks state from the end of one video into
        # the start of the next, which was observed to make results
        # non-deterministic between identical re-uploads of the same file.
        upload_pose_model = mp_pose.Pose(
            model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        try:
            while True:
                ret, frame = source.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                results = upload_pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                video_timestamp = frames_processed / fps
                frames_processed += 1
                if results.pose_landmarks:
                    frames_with_pose += 1
                    alignment_checker.push_frame(results.pose_landmarks)
                    trunk_checker.push_frame(results.pose_landmarks)
                    _session.process_frame(results.pose_landmarks, w, h, video_timestamp)
        finally:
            source.release()
            upload_pose_model.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    winning_track = _session._candidates[_session._decide_tracked_landmark_side()]
    _session.finalize_recording()
    side = "left" if _session.state_machine.state == RehabState.RECORD_LEFT else "right"
    limb = _session.state_machine.left if side == "left" else _session.state_machine.right

    return jsonify({
        "side": side,
        "detected_landmark_side": _session._detected_landmark_side,
        "frames_processed": frames_processed,
        "frames_with_pose": frames_with_pose,
        "peak_velocity_deg_s": limb.peak_velocity_deg_s,
        "peak_extension_angle_deg": limb.peak_extension_angle_deg,
        "extension_deficit_deg": limb.extension_deficit_deg,
        "is_inhibited": limb.is_inhibited,
        "reps": winning_track.reps,
        "camera_alignment": alignment_checker.result(),
        "trunk_compliance": trunk_checker.result(),
    })


@app.route("/api/rehab/advance", methods=["POST"])
@login_required
def advance_state():
    """Manually push the state machine forward (e.g. clinician/patient taps 'Next')."""
    new_state = _session.advance_state()
    return jsonify({"state": new_state.value})


@app.route("/api/rehab/report")
@login_required
def report():
    """Final clinical report — only meaningful once state has reached ANALYZE."""
    return jsonify(_session.final_report())


@app.route("/api/rehab/reset", methods=["POST"])
@login_required
def reset_session():
    global _session
    _session = RehabSession()
    return jsonify({"state": _session.state_machine.state.value})


# ─────────────────────────────────────────────────────────────────────────────
# 11. BATCH / OFFLINE VALIDATION MODE
# ─────────────────────────────────────────────────────────────────────────────
def run_batch_validation(left_video_path: str, right_video_path: str) -> dict:
    """
    Non-Flask entry point for automated regression testing: feeds a fixed
    left-leg clip and right-leg clip straight through the pipeline and
    returns the final LSI report, with no webcam or browser involved.
    """
    session = RehabSession()

    def _drain(video_path: str):
        cap = VideoSource(video_path)
        # Same reasoning as the /upload route: a file is processed faster than
        # real time, so the timestamp must come from the video's own frame
        # position (frame_index / fps), not wall-clock time.
        fps = cap.cap.get(cv2.CAP_PROP_FPS) or 30.0
        pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                video_timestamp = frame_idx / fps
                frame_idx += 1
                if results.pose_landmarks:
                    session.process_frame(results.pose_landmarks, w, h, video_timestamp)
        finally:
            cap.release()
            pose.close()

    session.advance_state()  # SELECT_LEFT -> RECORD_LEFT
    _drain(left_video_path)
    session.finalize_recording()
    session.advance_state()  # RECORD_LEFT -> PROMPT_SWITCH
    session.advance_state()  # PROMPT_SWITCH -> SELECT_RIGHT
    session.advance_state()  # SELECT_RIGHT -> RECORD_RIGHT
    _drain(right_video_path)
    session.finalize_recording()
    session.advance_state()  # RECORD_RIGHT -> ANALYZE

    return session.final_report()


if __name__ == "__main__":
    # ── MOCK/LIVE TOGGLE ─────────────────────────────────────────────────────
    # Set RUN_MODE to "live" to launch the Flask MJPEG server against a webcam,
    # or "batch" to run a headless validation pass against fixed test clips.
    RUN_MODE = os.environ.get("REHAB_RUN_MODE", "live")

    if RUN_MODE == "batch":
        TEST_VIDEO_PATHS = {
            "left": os.path.join(os.path.dirname(__file__), "data", "rehab", "left_knee_extension.mp4"),
            "right": os.path.join(os.path.dirname(__file__), "data", "rehab", "right_knee_extension.mp4"),
        }
        result = run_batch_validation(TEST_VIDEO_PATHS["left"], TEST_VIDEO_PATHS["right"])
        print(result)
    else:
        app.run(host="0.0.0.0", port=5050, debug=True)
