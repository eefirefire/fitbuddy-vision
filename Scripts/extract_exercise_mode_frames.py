"""
Extracts evenly-spaced frames from a clip exactly the way analyzeService.js's
extractTemporalFrameMatrices does in the browser (maxFrames evenly spaced
across duration, resized to 256x256 JPEG), so the Exercise-mode validation
test sees the same input the real app would have sent.
"""
import os
import sys

import cv2


def extract_frames(video_path: str, output_dir: str, max_frames: int = 15):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    interval = max(1, total_frames // max_frames)
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    saved = []

    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (256, 256))
        out_path = os.path.join(output_dir, f"{clip_name}_frame{len(saved):02d}.jpg")
        cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        saved.append(out_path)
        if len(saved) >= max_frames:
            break

    cap.release()
    return saved


if __name__ == "__main__":
    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    frames = extract_frames(video_path, output_dir)
    print(f"Extracted {len(frames)} frames to {output_dir}")
