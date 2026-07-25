"""
Performance benchmark for the FitBuddy-AI rehab pipeline.

Measures, per clip, against the REAL processing pipeline (MediaPipe pose +
RehabSession kinematics — the same code path /api/rehab/upload uses):
  - Achieved FPS (frames actually processed per second of wall-clock time)
  - Average per-frame latency (ms)
  - Pose-detection miss count/rate (frames where MediaPipe found no person)
  - Detected rep count vs. a manually-verified ground truth (where known)

This is a throughput/engineering benchmark, not a clinical accuracy test —
see test_rehab_knee_extension.py for the correctness regression suite.
"""
import os
import time

import cv2
import mediapipe as mp

from rehab_knee_extension import RehabSession, RehabState, mp_pose

DOWNLOADS = r"C:\Users\User\Downloads"

# Ground truth rep counts, manually verified by inspecting extracted frames.
# None = inspected but ambiguous/not confidently countable by eye (reported
# honestly as such, not guessed).
CLIPS = [
    {"file": "Recording 2026-06-23 143936.mp4", "ground_truth_reps": 3, "note": "clean 3-rep side-on clip, manually verified"},
    {"file": "Recording 2026-06-24 163935.mp4", "ground_truth_reps": None, "note": "short real footage + non-exercise cartoon outro contaminating later frames"},
    {"file": "Recording 2026-06-24 182432.mp4", "ground_truth_reps": None, "note": "very short clip (70 frames); extension visible but not clearly a full clean rep by eye"},
    {"file": "Recording 2026-06-24 182936.mp4", "ground_truth_reps": None, "note": "knee angle looks nearly static (~90-110 deg) across sampled frames; no confident visible rep"},
]


def benchmark_clip(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose_model = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    session = RehabSession()
    session.advance_state()  # SELECT_LEFT -> RECORD_LEFT, so process_frame is active

    frame_count = 0
    pose_hits = 0
    total_latency_s = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.perf_counter()
            h, w = frame.shape[:2]
            results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            video_timestamp = frame_count / fps_native
            if results.pose_landmarks:
                pose_hits += 1
                session.process_frame(results.pose_landmarks, w, h, video_timestamp)
            frame_latency_s = time.perf_counter() - frame_start

            total_latency_s += frame_latency_s
            frame_count += 1
    finally:
        cap.release()
        pose_model.close()

    session.finalize_recording()
    winning_side = session._decide_tracked_landmark_side()
    detected_reps = len(session._candidates[winning_side].reps)

    return {
        "frame_count": frame_count,
        "pose_hits": pose_hits,
        "pose_misses": frame_count - pose_hits,
        "total_latency_s": total_latency_s,
        "avg_latency_ms": (total_latency_s / frame_count * 1000) if frame_count else None,
        "achieved_fps": (frame_count / total_latency_s) if total_latency_s > 0 else None,
        "detected_reps": detected_reps,
    }


def main():
    print(f"{'Clip':45s} {'FPS':>7s} {'Latency(ms)':>12s} {'Frames':>8s} {'Misses':>7s} {'Miss%':>7s} {'DetReps':>8s} {'GroundTruth':>11s}")
    print("-" * 120)

    for clip in CLIPS:
        path = os.path.join(DOWNLOADS, clip["file"])
        if not os.path.exists(path):
            print(f"{clip['file']:45s}  (file not found, skipped)")
            continue

        result = benchmark_clip(path)
        if result is None:
            print(f"{clip['file']:45s}  (could not open)")
            continue

        miss_pct = (result["pose_misses"] / result["frame_count"] * 100) if result["frame_count"] else 0
        gt = clip["ground_truth_reps"]
        gt_str = str(gt) if gt is not None else "uncertain"

        print(f"{clip['file']:45s} {result['achieved_fps']:7.1f} {result['avg_latency_ms']:12.1f} "
              f"{result['frame_count']:8d} {result['pose_misses']:7d} {miss_pct:6.1f}% "
              f"{result['detected_reps']:8d} {gt_str:>11s}")
        print(f"  note: {clip['note']}")


if __name__ == "__main__":
    main()
