"""
FitBuddy-AI — Sit-to-Stand Rehab Module
========================================

Clinical context
-----------------
Sit-to-stand (rising unassisted from a chair) is the single most cited
functional-mobility test in geriatric physical therapy: difficulty rising is
a well-documented predictor of both falls and broader functional decline in
older adults, and repeated sit-to-stand is a core movement in the Otago
Exercise Program (OEP) — the CDC-promoted, evidence-based fall-prevention
program. This module's parent app already cites CDC STEADI for its age-65+
fall-risk screening (see rehab_auth.py); this ties the SAME tracking
capability back to that same fall-risk framework, rather than adding an
unrelated movement just to prove the pipeline generalizes.

Why this is a SEPARATE file from rehab_knee_extension.py, not an extension
of it: sit-to-stand is structurally a different exercise, not a variant of
knee extension.
  - It's BILATERAL — both legs rise together. There is no "select left leg,
    then right leg" flow, so none of the LSI (Limb Symmetry Index) state
    machine from the knee-extension module applies here.
  - It's a TWO-JOINT movement — a real "stand" requires BOTH the hip and the
    knee to extend together, not one joint in isolation. Peak detection here
    tracks a composite angle (the weaker of the two joints), not a single
    joint angle.
  - It shares no clinical rationale with the knee-extension work beyond the
    underlying signal-processing techniques (smoothing, outlier-resistant
    peak detection, jitter classification) and the camera-alignment
    requirement, which genuinely ARE exercise-agnostic. Those pieces are
    deliberately duplicated into this file (not imported from
    rehab_knee_extension.py) so this module can be read, run, and reasoned
    about entirely on its own — see CameraAlignmentChecker, CueChannel,
    TemporalSequenceBuffer, KinematicDerivativesEngine, AMIAnomalyClassifier
    below, each carried over with the SAME hardening this session found
    necessary for the knee-extension pipeline (median-filtered outlier
    rejection on the rep-detection floor, a bounded cue-emission history so
    rapid same-window cues can't silently clobber each other, a settling
    grace period, etc.) rather than the earlier, less-hardened versions of
    that logic.
  - Account/login/intake (rehab_auth.py) IS imported, not duplicated — that
    module is genuinely generic (a username/password system and a pre-
    exercise safety questionnaire) and has nothing exercise-specific in it.
    Duplicating it would mean two separate user databases, which is a
    maintenance and security liability, not a benefit.

Honesty about validation status — read this before citing any number this
module produces: every threshold below is either (a) carried over unchanged
from the knee-extension module's ALREADY-validated-against-real-footage
constants (smoothing windows, jitter tolerances, sustained-frame
requirements) as a reasonable starting point, or (b) a new constant specific
to sit-to-stand (IDEAL_STANDING_DEG, the composite-angle margins) chosen by
the same reasoning style used elsewhere in this app, NOT independently
validated against real sit-to-stand footage yet. This module has only been
tested against synthetic angle sequences (see test_rehab_sit_to_stand.py) as
of this writing. The first real-world check is the planned goniometer
validation session — report that as a small, honestly-labeled preliminary
check (e.g. "preliminary check, n=6"), the same tone the rest of this
project's validation work uses, not as a finished clinical result.
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
    Pure vector math — no MediaPipe, no OpenCV, no I/O. Generalized to any
    three landmarks (unlike the knee-extension module's knee-only version)
    because sit-to-stand needs the SAME dot-product angle formula at TWO
    different vertices: the knee (hip-knee-ankle) and the hip
    (shoulder-hip-knee). Duplicated here rather than imported so this module
    has no dependency on rehab_knee_extension.py.
    """

    @staticmethod
    def calculate_joint_angle(a_xy, vertex_xy, c_xy) -> float:
        """
        theta = arccos( (v1 . v2) / (|v1| |v2|) )

        v1 = vertex -> a, v2 = vertex -> c. Both vectors originate AT the
        vertex, which is what makes arccos return the true interior angle at
        that joint (see rehab_knee_extension.VectorGeometryEngine for the
        full derivation — identical math, just not hardcoded to hip/knee/ankle).
        """
        a = np.asarray(a_xy, dtype=np.float64)
        vertex = np.asarray(vertex_xy, dtype=np.float64)
        c = np.asarray(c_xy, dtype=np.float64)

        v1 = a - vertex
        v2 = c - vertex

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return float("nan")

        cosine = np.dot(v1, v2) / (n1 * n2)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEMPORAL SEQUENCE WINDOW BUFFER  (unchanged from rehab_knee_extension.py)
# ─────────────────────────────────────────────────────────────────────────────
class TemporalSequenceBuffer:
    """Rolling window of (timestamp, angle) samples — see
    rehab_knee_extension.TemporalSequenceBuffer for the full reasoning
    (~1.5s of history at 30fps, enough to see a full ascent/descent phase
    without holding unbounded memory)."""

    def __init__(self, maxlen: int = 45):
        self.maxlen = maxlen
        self._timestamps = deque(maxlen=maxlen)
        self._angles = deque(maxlen=maxlen)

    def push(self, timestamp: float, angle: float) -> None:
        if np.isnan(angle):
            return
        self._timestamps.append(timestamp)
        self._angles.append(angle)

    def as_arrays(self):
        return np.array(self._timestamps), np.array(self._angles)

    def is_ready(self, min_samples: int = 3) -> bool:
        return len(self._angles) >= min_samples

    def __len__(self):
        return len(self._angles)


# ─────────────────────────────────────────────────────────────────────────────
# 3. KINEMATIC DERIVATIVES ENGINE  (unchanged from rehab_knee_extension.py)
# ─────────────────────────────────────────────────────────────────────────────
class KinematicDerivativesEngine:
    """Differentiates the temporal buffer to angular velocity/acceleration
    using a trailing (causal) moving average before each derivative — see
    rehab_knee_extension.KinematicDerivativesEngine for why a centered
    average silently crushed the most recent (i.e. "current") sample and why
    that mattered. SMOOTHING_WINDOW=9 was tuned against real knee-extension
    footage; carried over here as a reasonable starting point, not
    re-validated against sit-to-stand footage yet."""

    SMOOTHING_WINDOW = 9

    @classmethod
    def _smooth(cls, values: np.ndarray) -> np.ndarray:
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
        return np.gradient(cls._smooth(angles), timestamps)

    @classmethod
    def compute_acceleration(cls, timestamps: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        if len(velocity) < 2:
            return np.zeros_like(velocity)
        return np.gradient(cls._smooth(velocity), timestamps)


# ─────────────────────────────────────────────────────────────────────────────
# 4. JITTER / STUTTER ANOMALY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
class AMIAnomalyClassifier:
    """
    Flags a stop-start ("stuttering") movement pattern via acceleration
    sign-flips — see rehab_knee_extension.AMIAnomalyClassifier for the full
    derivation of why shape (not speed) is the right signal, and how
    OSCILLATION_FLIP_THRESHOLD/ZERO_ACCELERATION_EPS were derived from real
    footage.

    Reused here on the composite standing_angle signal (see
    _CandidateBodySideTrack below) rather than re-derived for sit-to-stand
    specifically: the underlying technique — detecting a jerky vs. smooth
    motion from its acceleration trace — is a general signal-processing
    method, not knee-specific physiology. The clinical LABEL this produces
    for knee extension ("Arthrogenic Muscle Inhibition / pain-guarding") is
    NOT claimed here; for sit-to-stand this is presented only as "the rise
    wasn't smooth," a purely descriptive movement-quality flag, since a
    stuttering stand could reflect balance hesitation, weakness, or pain —
    this pipeline cannot and does not try to distinguish between those.
    """

    OSCILLATION_FLIP_THRESHOLD = 3
    ZERO_ACCELERATION_EPS = 100.0

    @classmethod
    def _count_acceleration_sign_flips(cls, acceleration: np.ndarray) -> int:
        eps = cls.ZERO_ACCELERATION_EPS
        signs = np.sign(np.where(np.abs(acceleration) < eps, 0.0, acceleration))
        signs = signs[signs != 0]
        if len(signs) < 2:
            return 0
        return int(np.sum(np.diff(signs) != 0))

    @classmethod
    def classify_motion(cls, velocity: np.ndarray, acceleration: np.ndarray) -> dict:
        if len(velocity) == 0:
            return {"peak_velocity_deg_s": 0.0, "oscillation_flips": 0, "is_jerky": False}
        peak_velocity = float(np.max(velocity))
        flips = cls._count_acceleration_sign_flips(acceleration)
        return {
            "peak_velocity_deg_s": round(peak_velocity, 1),
            "oscillation_flips": flips,
            "is_jerky": flips >= cls.OSCILLATION_FLIP_THRESHOLD,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. CUE CHANNEL  (unchanged from rehab_knee_extension.py — see that file's
#    CueChannel docstring for why history/pending_since exists: two cues in
#    one category can legitimately fire within one poll interval, and a
#    single {text, seq} slot silently drops whichever fired first)
# ─────────────────────────────────────────────────────────────────────────────
class CueChannel:
    HISTORY_LIMIT = 10

    def __init__(self):
        self.text: str | None = None
        self.seq: int = 0
        self.history: list[tuple[int, str]] = []

    def emit(self, text: str) -> None:
        self.text = text
        self.seq += 1
        self.history.append((self.seq, text))
        if len(self.history) > self.HISTORY_LIMIT:
            self.history = self.history[-self.HISTORY_LIMIT:]

    def pending_since(self, last_seen_seq: int) -> list[dict]:
        return [{"seq": seq, "text": text} for seq, text in self.history if seq > last_seen_seq]


# ─────────────────────────────────────────────────────────────────────────────
# 6. STANDING-EXTENSION DEFICIT CAPTURE
# ─────────────────────────────────────────────────────────────────────────────
class StandTransitionTracker:
    """
    Sit-to-stand's equivalent of rehab_knee_extension.ExtensionDeficitTracker
    — tracks the running highest COMPOSITE angle reached since the person
    last left the seated position, commits it as the rep's peak once the
    angle genuinely descends away from it. See that class's docstring for
    why "genuinely descends," not "velocity near zero," is what triggers a
    commit — same reasoning applies unchanged here.

    The key difference from knee extension: what's being tracked is a
    COMPOSITE angle (see _CandidateBodySideTrack.push_frame — the minimum of
    hip angle and knee angle), not a single joint. A real "stand" requires
    BOTH joints to extend; taking the minimum means a person who locks their
    knee but doesn't extend their hip (a common compensation — pushing up
    through the legs without straightening the torso) does NOT get credited
    with a full stand, which is exactly the distinction a real sit-to-stand
    assessment cares about.
    """

    # 180.0 is the anatomical maximum this vector-geometry formula can ever
    # return, not a claim about what "standing" clinically requires. Older
    # adults commonly don't reach full anatomical hip/knee extension even
    # when standing normally (mild flexion contracture, prior joint
    # replacement, or just normal aging stance), so — same reasoning as
    # rehab_knee_extension.ExtensionDeficitTracker.IDEAL_FULL_EXTENSION_DEG —
    # this targets a realistic "functionally standing" bar for this app's
    # older-adult audience rather than a perfectly straight anatomical
    # lockout. This is a deliberate product choice, not a specific cited
    # clinical cutoff.
    IDEAL_STANDING_DEG = 165.0

    # Same purpose as rehab_knee_extension's REARM_MARGIN_DEG: the person
    # must visibly return toward their own observed seated baseline before
    # the tracker looks for the next rep's peak, so jitter at the bottom of
    # a rep (or a momentary noise dip mid-rise) can't get double-counted or
    # phantom-counted. Relative to each person's OWN observed minimum, not a
    # fixed absolute number — different chair heights and camera framing
    # shift where "seated" sits in the frame.
    REARM_MARGIN_DEG = 15.0

    # Same purpose as MIN_PEAK_MARGIN_DEG in the knee-extension module: how
    # far above this person's own observed seated baseline the composite
    # angle must climb before a rise counts as a genuine rep attempt, rather
    # than requiring a fixed absolute angle regardless of individual mobility.
    MIN_PEAK_MARGIN_DEG = 25.0

    # How far the composite angle must drop below its running max before
    # that max is treated as final and committed as the rep's peak.
    DESCENT_MARGIN_DEG = 8.0

    # Minimum acceptable duration for the eccentric (lowering/sitting-back-
    # down) phase. Sit-to-stand's own literature (unlike knee extension)
    # DOES commonly time this phase directly — many clinical sit-to-stand
    # protocols use a controlled multi-second descent as part of assessing
    # control, not just concentric strength — but the exact figure below is
    # still carried over from the knee-extension module's ACSM-tempo-derived
    # number rather than a sit-to-stand-specific source, so treat it with
    # the same "derived proxy, not a validated cutoff for this movement"
    # caveat that module states explicitly.
    MIN_ECCENTRIC_DURATION_S = 2.0

    # Live-voice-cue-only threshold, more lenient than
    # MIN_ECCENTRIC_DURATION_S for the same reason as the knee-extension
    # module's LIVE_CUE_FAST_DESCENT_S: only cue live if the sit-down looks
    # like an actual uncontrolled drop, not merely brisker than a written
    # report's stricter tempo standard.
    LIVE_CUE_FAST_DESCENT_S = 1.0

    # Median-filter window applied to the raw composite angle before it's
    # allowed to lower the rep-acceptance floor — see
    # rehab_knee_extension.ExtensionDeficitTracker.MIN_ANGLE_MEDIAN_WINDOW
    # for why this replaced an earlier streak-based confirmation scheme
    # (that scheme could stall peak detection entirely against normal-speed
    # movement; a rolling median has no such stall condition).
    MIN_ANGLE_MEDIAN_WINDOW = 3

    def __init__(self):
        self._armed = True
        self._running_max_angle = float("-inf")
        self._observed_min_angle = float("inf")
        self._recent_angles = deque(maxlen=self.MIN_ANGLE_MEDIAN_WINDOW)
        self._peak_timestamp = None
        self._baseline_samples: list[float] = []

    def _update_observed_min(self, current_angle: float) -> None:
        self._recent_angles.append(current_angle)
        smoothed = float(np.median(self._recent_angles))
        self._observed_min_angle = min(self._observed_min_angle, smoothed)

    def observe_baseline(self, angle: float) -> None:
        """Call during the live settling grace period instead of
        check_for_peak — see RehabSitToStandSession.is_settling /
        LIVE_SETUP_GRACE_PERIOD_S. Accumulates samples for finalize_baseline
        to summarize once settling ends -- does NOT feed _observed_min_angle
        directly. A running MIN (what this used to do) is the least robust
        statistic possible against a contaminated settling window: found
        against real footage (sit_to_stand_controlled.mp4) that the video's
        own opening ~0.3-0.5s reads a spurious ~18 degree low (camera
        panning in / subject not yet framed) before settling into the
        subject's real range -- a running min would permanently anchor the
        entire session's rep-detection floor to that one bad artifact."""
        if np.isnan(angle):
            return
        self._baseline_samples.append(angle)

    def finalize_baseline(self) -> None:
        """Call exactly once, when settling ends and real rep tracking is
        about to begin. Summarizes observe_baseline's accumulated samples
        with a MEDIAN rather than a min -- robust to a handful of
        contaminated opening frames, since it takes a majority of bad
        samples (not just one) to meaningfully move a median. If settling
        never ran (no samples), leaves _observed_min_angle at its default
        (inf), preserving the old unbounded-fallback behavior."""
        if self._baseline_samples:
            self._observed_min_angle = float(np.median(self._baseline_samples))

    def check_for_peak(self, current_angle: float, timestamp: float = None):
        """
        Returns (peak_angle, standing_deficit, eccentric_duration_s) — same
        three-way return shape and timing semantics as
        rehab_knee_extension.ExtensionDeficitTracker.check_for_peak; see
        that method's docstring for the full contract.

        Callers are expected to have already screened out implausible
        readings (see _CandidateBodySideTrack.push_frame's rejection guard)
        — this method trusts current_angle at face value and has no
        amplitude sanity-checking of its own.
        """
        self._update_observed_min(current_angle)

        eccentric_duration_s = None
        if current_angle <= self._observed_min_angle + self.REARM_MARGIN_DEG:
            if self._peak_timestamp is not None and timestamp is not None:
                eccentric_duration_s = round(float(timestamp) - float(self._peak_timestamp), 2)
            self._peak_timestamp = None
            self._armed = True
            self._running_max_angle = float("-inf")

        if self._armed:
            self._running_max_angle = max(self._running_max_angle, current_angle)

        descended = (self._running_max_angle - current_angle) >= self.DESCENT_MARGIN_DEG
        peak_detected = (
            self._armed
            and descended
            and self._running_max_angle >= (self._observed_min_angle + self.MIN_PEAK_MARGIN_DEG)
        )

        if not peak_detected:
            return None, None, eccentric_duration_s

        self._armed = False
        self._peak_timestamp = timestamp
        peak_angle = self._running_max_angle
        # Clamped to >=0 for the same reason as rehab_knee_extension's
        # deficit clamp: IDEAL_STANDING_DEG (165) is below the anatomical
        # max (180), so a genuinely tall/mobile stand can legitimately
        # exceed the target, and a negative "deficit" is meaningless — a
        # peak beyond the target is still just a full stand, not a bonus.
        deficit = max(round(self.IDEAL_STANDING_DEG - peak_angle, 1), 0.0)
        return round(peak_angle, 1), deficit, eccentric_duration_s


# ─────────────────────────────────────────────────────────────────────────────
# 7. POSE LANDMARK EXTRACTION (MediaPipe)
# ─────────────────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Side-profile landmark indices needed per candidate side. Includes SHOULDER
# (not needed by the knee-extension module) because computing the hip angle
# needs a torso vector (shoulder->hip), not just the leg.
_LANDMARK_SETS = {
    "left": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP,
             mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    "right": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP,
              mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
}


def extract_body_landmarks(pose_landmarks, side: str, frame_w: int, frame_h: int):
    """Pulls 2D pixel-space (shoulder, hip, knee, ankle) for the given side —
    see rehab_knee_extension.extract_hip_knee_ankle for the scaling
    rationale (pixel-space is for on-screen overlay alignment only; the
    angle math itself is scale-invariant)."""
    shoulder_lm, hip_lm, knee_lm, ankle_lm = _LANDMARK_SETS[side]
    lm = pose_landmarks.landmark

    def to_xy(landmark_enum):
        p = lm[landmark_enum.value]
        return (p.x * frame_w, p.y * frame_h)

    return to_xy(shoulder_lm), to_xy(hip_lm), to_xy(knee_lm), to_xy(ankle_lm)


# Below this MediaPipe landmark visibility score, a joint reading is
# rejected outright (fully absent/occluded landmark). Deliberately lenient:
# checked against real footage (sit_to_stand_controlled.mp4) and found that
# ankle visibility normally sits around 0.3-0.5 throughout most of that
# video just from ordinary camera framing (feet near the frame edge), with
# no meaningful dip during the actual tracking glitch this module's other
# defense (MAX_PLAUSIBLE_VELOCITY_DEG_S, see _CandidateBodySideTrack) was
# built to catch -- MediaPipe reported moderate-to-high confidence on every
# landmark even while producing a physically impossible angle. So this gate
# only screens out genuinely-absent detections, not the low-but-usable
# confidence that's normal in real footage; it is not the primary defense.
MIN_LANDMARK_VISIBILITY = 0.15


def landmarks_confident(pose_landmarks, side: str) -> bool:
    lm = pose_landmarks.landmark
    return all(lm[l.value].visibility >= MIN_LANDMARK_VISIBILITY for l in _LANDMARK_SETS[side])


class CameraAlignmentChecker:
    """
    Unchanged from rehab_knee_extension.py — detects whether the camera is
    filming a clean side profile, using the same shoulder/hip horizontal-
    separation-vs-torso-height ratio. This check is exercise-agnostic: ANY
    2D single-camera angle measurement (knee extension, sit-to-stand, or
    anything else this app might add later) assumes a sagittal-plane view,
    so the same thresholds and the same recent_level()/result() split apply
    unchanged. See that file's CameraAlignmentChecker docstring for the
    empirical derivation of GOOD_MAX_RATIO/POOR_MIN_RATIO, and
    recent_level()'s docstring for why live cueing uses a recent window
    instead of a whole-session average.
    """

    GOOD_MAX_RATIO = 0.25
    POOR_MIN_RATIO = 0.40
    RECENT_WINDOW = 30

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
            return

        shoulder_width = abs(ls.x - rs.x)
        hip_width = abs(lh.x - rh.x)
        ratio = (shoulder_width + hip_width) / 2 / torso_height
        self._ratios.append(ratio)

    def _bucket(self, mean_ratio: float) -> tuple[str, str | None]:
        if mean_ratio < self.GOOD_MAX_RATIO:
            return "good", None
        if mean_ratio < self.POOR_MIN_RATIO:
            return "caution", (
                "Your camera angle looks slightly off from a true side profile. "
                "Try squaring the camera up to your side before your next clip."
            )
        return "poor", (
            "Your camera doesn't look like it's filming a side profile. "
            "Reposition the camera directly to your side and re-record."
        )

    def result(self) -> dict:
        """Whole-session average — for the post-session report."""
        if not self._ratios:
            return {"ratio": None, "level": "unknown", "message": None}
        mean_ratio = float(np.mean(self._ratios))
        level, message = self._bucket(mean_ratio)
        return {"ratio": round(mean_ratio, 3), "level": level, "message": message}

    def recent_level(self) -> str:
        """Recent-window average — for live cueing. See
        rehab_knee_extension.CameraAlignmentChecker.recent_level for why a
        whole-session average is the wrong signal for a live indicator."""
        if not self._ratios:
            return "unknown"
        recent = self._ratios[-self.RECENT_WINDOW:]
        level, _ = self._bucket(float(np.mean(recent)))
        return level


# ─────────────────────────────────────────────────────────────────────────────
# 8. PER-CANDIDATE BODY-SIDE TRACK
# ─────────────────────────────────────────────────────────────────────────────
class _CandidateBodySideTrack:
    """
    Independent pipeline for ONE candidate MediaPipe landmark side ('left' or
    'right'). RehabSitToStandSession runs two of these in parallel — same
    occlusion-resistant reasoning as
    rehab_knee_extension._CandidateLegTrack: the side facing away from the
    camera is occluded but MediaPipe still emits a confident-looking guess
    for it, so both sides are tracked and the one that actually shows real
    motion wins (see range_of_motion / RehabSitToStandSession.
    _decide_tracked_landmark_side).

    Unlike the knee-extension module, "which side" here is about which body
    side is camera-facing, NOT which leg is being exercised — sit-to-stand
    is bilateral, both legs rise together, so there's no per-leg selection
    flow, just a single continuous recording.
    """

    SUSTAINED_FRAMES_REQUIRED = 5  # see rehab_knee_extension for derivation

    # AMIAnomalyClassifier's oscillation-flip signal is shape-only, not
    # amplitude-aware -- it was validated against short knee-extension clips
    # where "between reps" was a brief cut, never a real held-still pause.
    # Sit-to-stand sessions run this classifier continuously across an
    # entire recording, INCLUDING long rest/talking pauses between reps
    # (confirmed against real footage: sit_to_stand_controlled.mp4 has
    # 13-97s gaps between reps where a trainer stands still narrating).
    # During those gaps, sub-pixel MediaPipe landmark jitter on an otherwise
    # motionless person still produces enough acceleration sign-flips to
    # read as "jerky" after double differentiation, firing a false "Slow
    # down" cue for someone who isn't moving at all. Every genuine rep
    # across all three real test videos had a peak velocity of at least
    # ~74 deg/s (the slowest observed full stand); this floor is set well
    # below that and well above plausible landmark-jitter noise, so it
    # gates jerky classification to windows with real physical motion
    # without needing to touch the shared, already-validated classifier.
    MIN_VELOCITY_FOR_JERKY_DEG_S = 20.0

    # Above this implied RAW (unsmoothed) frame-to-frame velocity, a single
    # reading is rejected outright and never reaches angle history, the AMI
    # classifier, or StandTransitionTracker. Deliberately a backup defense,
    # not the primary one -- MAX_DRIFT_BELOW_SETTLED_BASELINE_DEG below is
    # what actually bounds the real-world glitch this module was hardened
    # against: that glitch's own internal frame-to-frame deltas are each
    # individually unremarkable (a gradual few-frame collapse, not one
    # teleport), so a velocity clamp alone cannot catch it -- confirmed
    # empirically: an earlier, tighter version of this constant (700 deg/s)
    # let the glitch through anyway while ALSO clipping a genuine fast rise
    # (this same video's real first rep has raw frame-to-frame deltas of
    # 900-1300+ deg/s -- well above what the classifier's SMOOTHED
    # peak-velocity metric reports, since smoothing dampens instantaneous
    # spikes). This ceiling is set safely above that genuine range, so it
    # only catches single-frame teleports too extreme to be human under any
    # interpretation (the glitch's worst single jump measured ~3800 deg/s).
    MAX_PLAUSIBLE_VELOCITY_DEG_S = 2500.0

    # Primary defense against the real glitch found in sit_to_stand_
    # controlled.mp4: a MediaPipe tracking glitch mid-session swung the
    # composite angle down to ~1 degree for a handful of frames via a
    # gradual few-frame collapse (see MAX_PLAUSIBLE_VELOCITY_DEG_S above for
    # why a velocity clamp alone can't catch that shape of glitch). Any
    # reading more than this many degrees below the tracker's own
    # observed_min_angle (which, post-settling, reflects the person's real
    # calibrated seated baseline) is rejected outright -- not fed into
    # angle history, the AMI classifier, or StandTransitionTracker's
    # armed/rearm state at all. Set generously (matching the same
    # ballpark as MIN_PEAK_MARGIN_DEG/REARM_MARGIN_DEG on
    # StandTransitionTracker) so genuine per-person variation in how low
    # someone sits isn't mistaken for a glitch.
    MAX_DRIFT_BELOW_SETTLED_BASELINE_DEG = 20.0

    def __init__(self):
        self.buffer = TemporalSequenceBuffer(maxlen=45)
        self.transition_tracker = StandTransitionTracker()
        self.min_angle_seen = float("inf")
        self.max_angle_seen = float("-inf")
        self._angle_samples: list[float] = []
        self._last_accepted_angle: float | None = None
        self._last_accepted_timestamp: float | None = None
        # One entry per detected rep, each recording BOTH joint angles at
        # peak (not just the composite) — per the original build-scope note
        # for this feature ("tracking hip and knee angle at the top/bottom
        # of the movement"), since a clinician/goniometer comparison wants
        # the individual joint numbers, not just the combined pass/fail one.
        self.reps: list[dict] = []
        self._current_rep_peak_velocity = 0.0
        self._current_rep_jerky = False
        self._consecutive_jerky_frames = 0
        self.latest_hip_angle: float | None = None
        self.latest_knee_angle: float | None = None
        self.latest_standing_angle: float | None = None
        # This rep's hip/knee angle AT the composite peak instant — captured
        # alongside running_max_angle inside push_frame, since the composite
        # StandTransitionTracker only tracks the single combined number.
        self._hip_at_running_peak: float | None = None
        self._knee_at_running_peak: float | None = None
        self.cue = CueChannel()

    @property
    def range_of_motion(self) -> float:
        """Used to decide which candidate side is the real, camera-facing
        one — see rehab_knee_extension._CandidateLegTrack.range_of_motion
        for why a percentile range (not raw max-min) is used: a single
        one-frame jitter spike on the occluded side can otherwise look like
        more motion than the real side's genuine repeated cycles."""
        if len(self._angle_samples) < 5:
            if self.max_angle_seen == float("-inf"):
                return 0.0
            return self.max_angle_seen - self.min_angle_seen
        return float(np.percentile(self._angle_samples, 90) - np.percentile(self._angle_samples, 10))

    @property
    def best_rep(self) -> dict | None:
        if not self.reps:
            return None
        return max(self.reps, key=lambda r: r["standing_angle"])

    def observe_settling_angle(self, standing_angle: float) -> None:
        """Call during the settling grace period instead of push_frame —
        see StandTransitionTracker.observe_baseline."""
        if np.isnan(standing_angle):
            return
        self.transition_tracker.observe_baseline(standing_angle)

    def finalize_settling(self) -> None:
        """Call exactly once, when settling ends — see
        StandTransitionTracker.finalize_baseline."""
        self.transition_tracker.finalize_baseline()

    def push_frame(self, hip_angle: float, knee_angle: float, timestamp: float) -> dict:
        # A real stand needs BOTH joints extended — taking the minimum means
        # a partial compensation (e.g. knee locks out but hips stay flexed)
        # doesn't get credited as a full rep. See StandTransitionTracker's
        # docstring for why this matters clinically, not just numerically.
        standing_angle = min(hip_angle, knee_angle)

        # Reject a physically-implausible frame-to-frame jump before it can
        # reach ANY tracked state (angle history, the AMI classifier,
        # observed_min) — see MAX_PLAUSIBLE_VELOCITY_DEG_S above for why
        # this exists and why visibility gating alone doesn't catch it.
        if self._last_accepted_timestamp is not None:
            dt = timestamp - self._last_accepted_timestamp
            if dt > 0:
                implied_speed = abs(standing_angle - self._last_accepted_angle) / dt
                if implied_speed > self.MAX_PLAUSIBLE_VELOCITY_DEG_S:
                    return {"standing_angle": round(standing_angle, 1), "rejected_implausible_jump": True}

        # Second, independent guard: reject a reading implausibly far below
        # the calibrated seated baseline, even if it arrived via a gradual
        # multi-frame decline that never tripped the velocity guard above.
        # This is what actually stops the real glitch this module was
        # hardened against (see MAX_DRIFT_BELOW_SETTLED_BASELINE_DEG) --
        # confirmed necessary by testing: the velocity guard alone lets the
        # glitch's entry frames through (each individual frame-to-frame
        # step is unremarkable; only the cumulative ~10-frame trend is
        # implausible), and even bounding _observed_min_angle's own floor
        # wasn't sufficient on its own, because REARM compares the RAW
        # current angle against that floor -- an unclamped glitch value
        # still re-arms the tracker and lets ordinary standing sway
        # register as a spurious new rep right after the glitch passes.
        # Rejecting the frame entirely, before it reaches check_for_peak at
        # all, closes that gap in one place instead of chasing it through
        # every downstream comparison.
        observed_min = self.transition_tracker._observed_min_angle
        if observed_min != float("inf") and standing_angle < observed_min - self.MAX_DRIFT_BELOW_SETTLED_BASELINE_DEG:
            return {"standing_angle": round(standing_angle, 1), "rejected_implausible_low_angle": True}

        self._last_accepted_angle = standing_angle
        self._last_accepted_timestamp = timestamp

        self.latest_hip_angle = round(hip_angle, 1)
        self.latest_knee_angle = round(knee_angle, 1)
        self.latest_standing_angle = round(standing_angle, 1)
        self.min_angle_seen = min(self.min_angle_seen, standing_angle)
        self.max_angle_seen = max(self.max_angle_seen, standing_angle)
        self._angle_samples.append(standing_angle)

        # Track which joint reading corresponds to the CURRENT running-max
        # composite angle, so if this frame becomes the new peak we know
        # the real hip/knee numbers at that instant, not just the combined
        # minimum.
        if standing_angle >= self.transition_tracker._running_max_angle:
            self._hip_at_running_peak = hip_angle
            self._knee_at_running_peak = knee_angle

        self.buffer.push(timestamp, standing_angle)
        if not self.buffer.is_ready(min_samples=3):
            return {"standing_angle": round(standing_angle, 1)}

        timestamps, angles = self.buffer.as_arrays()
        velocity = KinematicDerivativesEngine.compute_velocity(timestamps, angles)
        acceleration = KinematicDerivativesEngine.compute_acceleration(timestamps, velocity)

        motion = AMIAnomalyClassifier.classify_motion(velocity, acceleration)
        current_velocity = float(velocity[-1])
        window_peak_speed = float(np.max(np.abs(velocity)))

        self._current_rep_peak_velocity = max(self._current_rep_peak_velocity, current_velocity)
        is_genuinely_jerky = motion["is_jerky"] and window_peak_speed >= self.MIN_VELOCITY_FOR_JERKY_DEG_S
        if is_genuinely_jerky:
            self._consecutive_jerky_frames += 1
        else:
            self._consecutive_jerky_frames = 0
        if self._consecutive_jerky_frames >= self.SUSTAINED_FRAMES_REQUIRED:
            was_already_jerky = self._current_rep_jerky
            self._current_rep_jerky = True
            if not was_already_jerky:
                self.cue.emit("Slow down")

        peak_angle, deficit, completed_eccentric_s = self.transition_tracker.check_for_peak(
            standing_angle, timestamp
        )
        if peak_angle is not None:
            self.reps.append({
                "rep_number": len(self.reps) + 1,
                "standing_angle": round(peak_angle, 1),
                "hip_angle_deg": round(self._hip_at_running_peak, 1) if self._hip_at_running_peak is not None else None,
                "knee_angle_deg": round(self._knee_at_running_peak, 1) if self._knee_at_running_peak is not None else None,
                "standing_deficit_deg": round(deficit, 1) if deficit is not None else 0.0,
                "peak_velocity_deg_s": round(self._current_rep_peak_velocity, 1),
                "is_jerky": self._current_rep_jerky,
                "eccentric_duration_s": None,
                "is_descent_too_fast": False,
            })
            if not self._current_rep_jerky:
                if deficit is not None and deficit > 15.0:
                    self.cue.emit("Stand up further")
                else:
                    self.cue.emit("Good")
            self._current_rep_peak_velocity = 0.0
            self._current_rep_jerky = False
            self._consecutive_jerky_frames = 0
            self._hip_at_running_peak = None
            self._knee_at_running_peak = None

        if completed_eccentric_s is not None and self.reps:
            self.reps[-1]["eccentric_duration_s"] = completed_eccentric_s
            is_descent_too_fast = bool(
                completed_eccentric_s < StandTransitionTracker.MIN_ECCENTRIC_DURATION_S
            )
            self.reps[-1]["is_descent_too_fast"] = is_descent_too_fast
            # Same live-vs-report threshold split, and same same-rep-dedup
            # guard, as rehab_knee_extension._CandidateLegTrack — see that
            # file for why: without the guard, a rep that's both jerky AND
            # fast-sitting could fire "Slow down" twice back to back.
            if completed_eccentric_s < StandTransitionTracker.LIVE_CUE_FAST_DESCENT_S and not self.reps[-1]["is_jerky"]:
                self.cue.emit("Slow down")

        return {
            "hip_angle_deg": round(hip_angle, 1),
            "knee_angle_deg": round(knee_angle, 1),
            "standing_angle": round(standing_angle, 1),
            "velocity_deg_s": round(current_velocity, 1),
            "motion": motion,
            "standing_deficit_deg": deficit,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 9. SESSION STATE MACHINE — simpler than knee extension's: sit-to-stand is
#    bilateral, so there's no per-leg select/switch flow and no LSI.
# ─────────────────────────────────────────────────────────────────────────────
class SitToStandState(enum.Enum):
    SELECT = "STATE_SELECT"
    RECORD = "STATE_RECORD"
    ANALYZE = "STATE_ANALYZE"


_STATE_ORDER = (SitToStandState.SELECT, SitToStandState.RECORD, SitToStandState.ANALYZE)


@dataclass
class SessionSummary:
    """Session-level summary once RECORD is finalized — sit-to-stand's
    equivalent of rehab_knee_extension.LimbRecord, but without any bilateral
    comparison since this exercise doesn't have a left/right split."""
    peak_velocity_deg_s: float = 0.0
    peak_standing_angle_deg: float = 0.0
    standing_deficit_deg: float = 0.0
    is_jerky: bool = False
    reps: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 10. SESSION ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
class RehabSitToStandSession:
    """One full sit-to-stand session — walks SELECT -> RECORD -> ANALYZE,
    feeding frames from a live webcam or video file, producing a live
    overlay and a final summary. See RehabSession in rehab_knee_extension.py
    for the closely-analogous (but bilateral/LSI) version this is modeled
    on."""

    LIVE_SETUP_GRACE_PERIOD_S = 3.0
    _ALIGNMENT_CUE_LEVELS = frozenset({"caution", "poor"})

    def __init__(self):
        self.state = SitToStandState.SELECT
        self._candidates = {"left": _CandidateBodySideTrack(), "right": _CandidateBodySideTrack()}
        self._detected_landmark_side = None
        self._alignment_checker = CameraAlignmentChecker()
        self._frames_processed = 0
        self._frames_with_pose = 0
        self._last_alignment_level = None
        self._live_recording_started_at = None
        self._settling_finalized = False
        self.alignment_cue = CueChannel()
        self.summary = SessionSummary()

    def _active(self) -> bool:
        return self.state == SitToStandState.RECORD

    def _reset_recording(self):
        self._candidates = {"left": _CandidateBodySideTrack(), "right": _CandidateBodySideTrack()}
        self._detected_landmark_side = None
        self._alignment_checker = CameraAlignmentChecker()
        self._frames_processed = 0
        self._frames_with_pose = 0
        self._last_alignment_level = None
        self._live_recording_started_at = None
        self._settling_finalized = False
        self.alignment_cue = CueChannel()

    def advance_state(self) -> SitToStandState:
        if self.state == SitToStandState.RECORD:
            self.finalize_recording()
        idx = _STATE_ORDER.index(self.state)
        if idx < len(_STATE_ORDER) - 1:
            self.state = _STATE_ORDER[idx + 1]
        if self.state == SitToStandState.RECORD:
            self._reset_recording()
        return self.state

    def finalize_settling(self) -> None:
        """Call exactly once, the moment settling ends and real rep
        tracking is about to begin — summarizes each candidate track's
        accumulated settling samples into its calibrated baseline (see
        StandTransitionTracker.finalize_baseline). Idempotent: safe to call
        even if settling never produced samples."""
        if self._settling_finalized:
            return
        self._settling_finalized = True
        for candidate in self._candidates.values():
            candidate.finalize_settling()

    def is_settling(self, now: float) -> bool:
        if self._live_recording_started_at is None:
            self._live_recording_started_at = now
        settling = (now - self._live_recording_started_at) < self.LIVE_SETUP_GRACE_PERIOD_S
        if not settling:
            self.finalize_settling()
        return settling

    def _decide_tracked_landmark_side(self) -> str:
        left_rom = self._candidates["left"].range_of_motion
        right_rom = self._candidates["right"].range_of_motion
        self._detected_landmark_side = "left" if left_rom >= right_rom else "right"
        return self._detected_landmark_side

    def process_frame(self, pose_landmarks, frame_w: int, frame_h: int, timestamp: float) -> dict:
        if not self._active() or pose_landmarks is None:
            return {}
        frame_results = {}
        for candidate in ("left", "right"):
            if not landmarks_confident(pose_landmarks, candidate):
                continue
            shoulder, hip, knee, ankle = extract_body_landmarks(pose_landmarks, candidate, frame_w, frame_h)
            hip_angle = VectorGeometryEngine.calculate_joint_angle(shoulder, hip, knee)
            knee_angle = VectorGeometryEngine.calculate_joint_angle(hip, knee, ankle)
            if np.isnan(hip_angle) or np.isnan(knee_angle):
                continue
            frame_results[candidate] = self._candidates[candidate].push_frame(hip_angle, knee_angle, timestamp)
        active_side = self._decide_tracked_landmark_side()
        return frame_results.get(active_side, {})

    def observe_settling_frame(self, pose_landmarks, frame_w: int, frame_h: int) -> None:
        if not self._active() or pose_landmarks is None:
            return
        for candidate in ("left", "right"):
            if not landmarks_confident(pose_landmarks, candidate):
                continue
            shoulder, hip, knee, ankle = extract_body_landmarks(pose_landmarks, candidate, frame_w, frame_h)
            hip_angle = VectorGeometryEngine.calculate_joint_angle(shoulder, hip, knee)
            knee_angle = VectorGeometryEngine.calculate_joint_angle(hip, knee, ankle)
            if np.isnan(hip_angle) or np.isnan(knee_angle):
                continue
            self._candidates[candidate].observe_settling_angle(min(hip_angle, knee_angle))

    def note_alignment_frame(self) -> None:
        if not self._active():
            return
        level = self._alignment_checker.recent_level()
        if level in self._ALIGNMENT_CUE_LEVELS and self._last_alignment_level not in self._ALIGNMENT_CUE_LEVELS:
            self.alignment_cue.emit("Check your camera angle")
        self._last_alignment_level = level

    def live_status(self, since_cue_seq: int = 0, since_alignment_seq: int = 0) -> dict:
        """Pollable snapshot for the browser — same shape/purpose as
        RehabSession.live_status in rehab_knee_extension.py. Side-effect-
        free; see that method's docstring for why (safe to call from both
        the frame loop and concurrent HTTP polling)."""
        if not self._active():
            return {"active": False, "state": self.state.value}

        settling = (
            self._live_recording_started_at is None
            or (time.time() - self._live_recording_started_at) < self.LIVE_SETUP_GRACE_PERIOD_S
        )

        track = self._candidates[self._decide_tracked_landmark_side()]
        alignment_level = self._alignment_checker.recent_level()

        observed_min = track.transition_tracker._observed_min_angle
        if observed_min == float("inf"):
            observed_min = track.latest_standing_angle if track.latest_standing_angle is not None else 90.0
        gauge_low = round(observed_min + StandTransitionTracker.MIN_PEAK_MARGIN_DEG, 1)
        gauge_low = min(gauge_low, StandTransitionTracker.IDEAL_STANDING_DEG - 1.0)

        return {
            "active": True,
            "settling": settling,
            "state": self.state.value,
            "movement": "sit_to_stand",
            "side": self._detected_landmark_side,
            "rep_count": len(track.reps),
            "angle": None if settling else track.latest_standing_angle,
            "hip_angle": None if settling else track.latest_hip_angle,
            "knee_angle": None if settling else track.latest_knee_angle,
            "target_range": [gauge_low, StandTransitionTracker.IDEAL_STANDING_DEG],
            "alignment": "unknown" if settling else alignment_level,
            "cue": None if settling else track.cue.text,
            "cue_seq": track.cue.seq,
            "cue_pending": [] if settling else track.cue.pending_since(since_cue_seq),
            "alignment_cue": None if settling else self.alignment_cue.text,
            "alignment_cue_seq": self.alignment_cue.seq,
            "alignment_cue_pending": [] if settling else self.alignment_cue.pending_since(since_alignment_seq),
        }

    def finalize_recording(self):
        if not self._active():
            return
        winner = self._decide_tracked_landmark_side()
        track = self._candidates[winner]
        best = track.best_rep
        self.summary = SessionSummary(
            peak_velocity_deg_s=best["peak_velocity_deg_s"] if best else 0.0,
            peak_standing_angle_deg=best["standing_angle"] if best else 0.0,
            standing_deficit_deg=best["standing_deficit_deg"] if best else 0.0,
            is_jerky=any(r["is_jerky"] for r in track.reps),
            reps=track.reps,
        )

    def final_report(self) -> dict:
        return {
            "state": self.state.value,
            "peak_velocity_deg_s": self.summary.peak_velocity_deg_s,
            "peak_standing_angle_deg": self.summary.peak_standing_angle_deg,
            "standing_deficit_deg": self.summary.standing_deficit_deg,
            "is_jerky": self.summary.is_jerky,
            "reps": self.summary.reps,
            "rep_count": len(self.summary.reps),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 11. VIDEO SOURCE  (unchanged from rehab_knee_extension.py)
# ─────────────────────────────────────────────────────────────────────────────
class VideoSource:
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
# 12. FLASK ROUTING / STATE-MACHINE LOOP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("REHAB_SECRET_KEY", "dev-only-fitbuddy-secret-change-me")
FRONTEND_ORIGIN = os.environ.get("REHAB_FRONTEND_ORIGIN", "http://localhost:5173")

init_db()
app.register_blueprint(auth_bp)


@app.after_request
def _allow_cors(response):
    response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# A single in-memory session for demo purposes — same caveat as
# rehab_knee_extension.py: a production deployment would key this by
# session/user id instead of a module-level global.
_session = RehabSitToStandSession()
_pose_model = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Own video source env var + own default port (5052) so this can run
# alongside rehab_knee_extension.py (5050) without conflict — deliberately
# NOT sharing state/process with that module, matching the "put everything
# into another file" instruction this module was built under.
VIDEO_SOURCE = os.environ.get("SIT_TO_STAND_VIDEO_SOURCE", "0")
try:
    VIDEO_SOURCE = int(VIDEO_SOURCE)
except ValueError:
    pass


def _draw_overlay(frame, frame_result: dict, state: SitToStandState, live: dict | None = None):
    cv2.putText(frame, f"STATE: {state.value}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if "hip_angle_deg" in frame_result:
        cv2.putText(frame, f"Hip Angle: {frame_result['hip_angle_deg']:.1f} deg", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    if "knee_angle_deg" in frame_result:
        cv2.putText(frame, f"Knee Angle: {frame_result['knee_angle_deg']:.1f} deg", (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    if "velocity_deg_s" in frame_result:
        cv2.putText(frame, f"Velocity: {frame_result['velocity_deg_s']:.1f} deg/s", (20, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    motion = frame_result.get("motion")
    if motion and motion.get("is_jerky"):
        cv2.putText(frame, "Uneven rise detected", (20, 144),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    if live and live.get("active"):
        h, w = frame.shape[:2]
        if live.get("settling"):
            msg = "GET READY..."
            (msg_w, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.putText(frame, msg, (w - msg_w - 20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 220, 0), 3, cv2.LINE_AA)
        else:
            rep_text = f"REPS: {live.get('rep_count', 0)}"
            (text_w, _), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            cv2.putText(frame, rep_text, (w - text_w - 20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

            cue = live.get("cue")
            if cue:
                (cue_w, _), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(frame, cue, (w - cue_w - 20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2, cv2.LINE_AA)

            if live.get("alignment") in ("caution", "poor"):
                warn = "CHECK CAMERA ANGLE"
                (warn_w, _), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(frame, warn, (w - warn_w - 20, 172),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return frame


def _mjpeg_generator():
    source = VideoSource(VIDEO_SOURCE)
    try:
        while True:
            ret, frame = source.read()
            if not ret:
                break

            # Bind the module global to a local once per frame — see
            # rehab_knee_extension._mjpeg_generator for why: with
            # threaded=True (needed so /live-status stays responsive during
            # this long-lived stream), a concurrent /reset can reassign the
            # global mid-iteration, and re-reading it on every statement
            # could split one frame's state across two session objects.
            session = _session

            h, w = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = _pose_model.process(image_rgb)

            frame_result = {}
            session._frames_processed += 1
            if results.pose_landmarks:
                session._frames_with_pose += 1
                now = time.time()
                session._alignment_checker.push_frame(results.pose_landmarks)
                if session._active():
                    if not session.is_settling(now):
                        session.note_alignment_frame()
                        frame_result = session.process_frame(results.pose_landmarks, w, h, now)
                    else:
                        session.observe_settling_frame(results.pose_landmarks, w, h)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            frame = _draw_overlay(frame, frame_result, session.state, session.live_status())

            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
    finally:
        source.release()


@app.route("/api/sit-to-stand/stream")
@login_required
def stream():
    """MJPEG live overlay stream — point an <img> tag at this endpoint."""
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/sit-to-stand/upload", methods=["POST"])
@login_required
def upload_video():
    if _session.state != SitToStandState.RECORD:
        return jsonify({
            "error": f"Cannot record a video while in state {_session.state.value}. "
                     "Click 'Next' to move into RECORD state first."
        }), 400

    uploaded = request.files.get("video")
    if uploaded is None:
        return jsonify({"error": "No video file in request (expected field 'video')."}), 400

    suffix = os.path.splitext(uploaded.filename or "clip.mp4")[1] or ".mp4"
    tmp_path = os.path.join(tempfile.gettempdir(), f"sit_to_stand_{uuid.uuid4().hex}{suffix}")
    uploaded.save(tmp_path)

    frames_processed = 0
    frames_with_pose = 0
    alignment_checker = CameraAlignmentChecker()
    try:
        source = VideoSource(tmp_path)
        fps = source.cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Fresh Pose instance for this upload, not the shared module-level
        # one — see rehab_knee_extension.upload_video for why (MediaPipe
        # keeps internal temporal state between .process() calls, and
        # reusing one instance across unrelated clips leaks state between
        # them).
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
                    _session.process_frame(results.pose_landmarks, w, h, video_timestamp)
        finally:
            source.release()
            upload_pose_model.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    winning_track = _session._candidates[_session._decide_tracked_landmark_side()]
    _session.finalize_recording()

    return jsonify({
        "side": _session._detected_landmark_side,
        "frames_processed": frames_processed,
        "frames_with_pose": frames_with_pose,
        "rep_count": len(winning_track.reps),
        "reps": winning_track.reps,
        "camera_alignment": alignment_checker.result(),
    })


@app.route("/api/sit-to-stand/clip-summary", methods=["POST"])
@login_required
def clip_summary():
    """Same purpose as rehab_knee_extension.clip_summary — call after
    stopping a live recording (before advancing state) to get the same
    feedback the upload endpoint returns."""
    if _session.state != SitToStandState.RECORD:
        return jsonify({"error": "No active recording to summarise."}), 400

    winning_track = _session._candidates[_session._decide_tracked_landmark_side()]
    _session.finalize_recording()

    return jsonify({
        "side": _session._detected_landmark_side,
        "frames_processed": _session._frames_processed,
        "frames_with_pose": _session._frames_with_pose,
        "rep_count": len(winning_track.reps),
        "reps": winning_track.reps,
        "camera_alignment": _session._alignment_checker.result(),
    })


@app.route("/api/sit-to-stand/advance", methods=["POST"])
@login_required
def advance_state():
    new_state = _session.advance_state()
    return jsonify({"state": new_state.value})


@app.route("/api/sit-to-stand/report")
@login_required
def report():
    return jsonify(_session.final_report())


@app.route("/api/sit-to-stand/reset", methods=["POST"])
@login_required
def reset_session():
    global _session
    _session = RehabSitToStandSession()
    return jsonify({"state": _session.state.value})


@app.route("/api/sit-to-stand/live-status")
@login_required
def live_status():
    since_cue_seq = request.args.get("since_cue_seq", default=0, type=int)
    since_alignment_seq = request.args.get("since_alignment_seq", default=0, type=int)
    return jsonify(_session.live_status(since_cue_seq, since_alignment_seq))


# ─────────────────────────────────────────────────────────────────────────────
# 13. BATCH / OFFLINE VALIDATION MODE
# ─────────────────────────────────────────────────────────────────────────────
def run_batch_validation(video_path: str) -> dict:
    """
    Non-Flask entry point for the goniometer validation session: feeds one
    recorded clip straight through the pipeline and returns every detected
    rep's hip/knee/composite peak angles, for direct comparison against
    manual goniometer readings taken during the same session — the same
    method rehab_knee_extension.run_batch_validation uses, and the same
    method the original knee-extension study used.

    Sit-to-stand is bilateral (no left/right split), so unlike the knee-
    extension version this only needs ONE video, not two.

    Runs a short settling/calibration window before rep tracking starts,
    using video-relative time instead of wall-clock time -- the same idea
    as a live session's RehabSitToStandSession.LIVE_SETUP_GRACE_PERIOD_S,
    but deliberately much shorter (BATCH_SETTLING_WINDOW_S below), not that
    same 3.0s constant reused as-is. This was missing until real footage
    testing found it necessary: without ANY calibration, the very first
    frames of a clip (camera panning in, subject not yet fully framed) seed
    StandTransitionTracker._observed_min_angle -- a running minimum that
    never resets for the life of the session -- with whatever garbage those
    opening frames show, corrupting rep detection for the entire rest of
    the video. Confirmed via sit_to_stand_controlled.mp4: the first ~0.3s
    reads a spurious ~18 degree low before the subject settles into their
    real range, which without this fix anchors the floor for the whole
    ~275s clip. The live grace period (3.0s) is deliberately NOT reused
    here: that constant assumes a live user still getting into position,
    but a pre-recorded clip's real exercise can start almost immediately
    (confirmed: this same clip's first genuine rep peaks at ~2.4s) -- a
    3.0s window would silently discard real reps that happen to start
    early, not just skip transient opening noise.
    """
    BATCH_SETTLING_WINDOW_S = 0.5
    session = RehabSitToStandSession()

    def _drain(path: str):
        cap = VideoSource(path)
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
                    if video_timestamp < BATCH_SETTLING_WINDOW_S:
                        session.observe_settling_frame(results.pose_landmarks, w, h)
                    else:
                        session.finalize_settling()
                        session.process_frame(results.pose_landmarks, w, h, video_timestamp)
        finally:
            cap.release()
            pose.close()

    session.advance_state()  # SELECT -> RECORD
    _drain(video_path)
    session.finalize_recording()
    session.advance_state()  # RECORD -> ANALYZE

    return session.final_report()


if __name__ == "__main__":
    RUN_MODE = os.environ.get("SIT_TO_STAND_RUN_MODE", "live")

    if RUN_MODE == "batch":
        TEST_VIDEO_PATH = os.path.join(
            os.path.dirname(__file__), "data", "sit_to_stand", "sit_to_stand_test.mp4"
        )
        result = run_batch_validation(TEST_VIDEO_PATH)
        print(result)
    else:
        # Default port 5052 -- deliberately different from
        # rehab_knee_extension.py's 5050, so both can run at once if needed.
        app.run(host="0.0.0.0", port=5052, debug=True, threaded=True)
