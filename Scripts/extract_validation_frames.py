"""
Exports the EXACT frame the AI pipeline used as each rep's peak, as a .jpg
you can upload to onlineprotractortool.com. Without this, you'd be
screenshotting whatever frame you eyeball as "the peak" while scrubbing the
video — which is rarely the same instant the AI actually measured, making
the angle_error_deg comparison apples-to-oranges. This runs the same
detection classes the real pipeline uses (ExtensionDeficitTracker via
_CandidateLegTrack), so the exported frame is the literal frame the
peak_angle in the AI output came from.

Usage:
    python extract_validation_frames.py "<path to clip>" [output_dir]
"""
import os
import sys

import cv2

from rehab_knee_extension import _CandidateLegTrack, VectorGeometryEngine, extract_hip_knee_ankle, mp_pose


def extract_validation_frames(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    candidates = {"left": _CandidateLegTrack(), "right": _CandidateLegTrack()}
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    saved = []

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                timestamp = frame_idx / fps
                for side, track in candidates.items():
                    before_count = len(track.reps)
                    hip, knee, ankle = extract_hip_knee_ankle(results.pose_landmarks, side, w, h)
                    angle = VectorGeometryEngine.calculate_knee_angle(hip, knee, ankle)
                    if angle == angle:  # filters NaN without importing math/numpy here
                        track.push_frame(angle, timestamp)
                    if len(track.reps) > before_count:
                        rep = track.reps[-1]
                        out_name = f"{clip_name}_{side}_rep{rep['rep_number']}_frame{frame_idx}_angle{rep['peak_angle']}.jpg"
                        out_path = os.path.join(output_dir, out_name)
                        cv2.imwrite(out_path, frame)
                        saved.append({
                            "side": side, "rep_number": rep["rep_number"],
                            "frame_number": frame_idx, "ai_peak_angle_deg": rep["peak_angle"],
                            "file": out_path,
                        })
            frame_idx += 1
    finally:
        cap.release()
        pose.close()

    # Same decision rule as RehabSession._decide_tracked_landmark_side: the
    # side with the larger percentile range of motion is the real, moving
    # leg. The OTHER side's entries (if any) are landmark-jitter artifacts
    # on the occluded leg, not real reps — flagging this explicitly avoids
    # the ambiguity of just printing every candidate with no indication of
    # which one the actual app would have committed.
    winning_side = "left" if candidates["left"].range_of_motion >= candidates["right"].range_of_motion else "right"
    return saved, winning_side


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_validation_frames.py <video_path> [output_dir]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "validation_frames"
    )

    results, winning_side = extract_validation_frames(video_path, output_dir)
    winning_results = [r for r in results if r["side"] == winning_side]
    other_results = [r for r in results if r["side"] != winning_side]

    print(f"Saved {len(results)} candidate peak frames to: {output_dir}")
    print(f"Winning side (use these for ground-truth measurement): {winning_side}\n")

    print(f"USE THESE — {len(winning_results)} rep(s) on the winning '{winning_side}' side:")
    for r in winning_results:
        print(f"  rep={r['rep_number']} frame={r['frame_number']:5d} "
              f"ai_angle={r['ai_peak_angle_deg']}  ->  {os.path.basename(r['file'])}")

    if other_results:
        print(f"\nIGNORE — {len(other_results)} likely landmark-jitter artifact(s) on the losing side:")
        for r in other_results:
            print(f"  side={r['side']:5s} rep={r['rep_number']} frame={r['frame_number']:5d} "
                  f"ai_angle={r['ai_peak_angle_deg']}  ->  {os.path.basename(r['file'])}")
