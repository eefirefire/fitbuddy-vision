import cv2
import uvicorn
import os
import shutil
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from math_utils import get_angle, get_vertical_angle

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

TEMP_DIR = "temp_videos"
os.makedirs(TEMP_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=TEMP_DIR), name="videos")

model = YOLO('yolov8n-pose.pt')

try:
    with open('fitbuddy_angle_brain.pkl', 'rb') as f:
        artifact = pickle.load(f)
        squat_classifier = artifact["model"]
        squat_features = artifact["features"]
except:
    squat_classifier = None

SQUAT_LABELS = {
    0: "Perfect Form", 1: "Too Shallow", 2: "Forward Lean", 
    3: "Knee Cave", 4: "Heel Lift", 5: "Uneven Balance"
}

class QualityGate:
    def __init__(self):
        self.bad_vis_frames = 0
        self.bad_track_frames = 0
        self.bad_light_frames = 0
        self.bad_blur_frames = 0
        self.small_person_frames = 0
        self.total_frames = 0
        self.prev_kps = None
        
        self.VISIBILITY_THRESH = 0.5
        self.JITTER_THRESH = 50.0
        self.BRIGHTNESS_THRESH = 40.0
        self.SIZE_THRESH = 0.3
        
        # 1. RE-ENABLE THE THRESHOLD
        # 35.0 is a standard value for detecting "soft" or out-of-focus images.
        self.BLUR_THRESH = 35.0 

    def check_frame(self, frame, keypoints, box, h, w):
        self.total_frames += 1
        issues = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness
        if cv2.Laplacian(gray, cv2.CV_64F).var() < self.BLUR_THRESH:
            self.bad_blur_frames += 1
            issues.append("Blurry")

        if np.mean(frame) < self.BRIGHTNESS_THRESH:
            self.bad_light_frames += 1
            issues.append("Too Dark")

        if keypoints is not None:
            confs = keypoints.conf[0].cpu().numpy()
            left_leg_vis = all(confs[i] > self.VISIBILITY_THRESH for i in [11, 13, 15])
            right_leg_vis = all(confs[i] > self.VISIBILITY_THRESH for i in [12, 14, 16])
            
            if not (left_leg_vis or right_leg_vis):
                self.bad_vis_frames += 1
                issues.append("Legs Hidden")

            xy = keypoints.xy[0].cpu().numpy()
            if self.prev_kps is not None:
                dist = np.linalg.norm(xy[[11,12,13,14]] - self.prev_kps[[11,12,13,14]], axis=1)
                if np.any(dist > self.JITTER_THRESH):
                    self.bad_track_frames += 1
                    issues.append("Jittery")
            self.prev_kps = xy

        if box is not None:
            _, _, _, box_h = box.xywh[0].cpu().numpy()
            if box_h / h < self.SIZE_THRESH:
                self.small_person_frames += 1
                issues.append("Too Small")

        return issues

    def get_verdict(self):
        verdict = {"passed": True, "errors": []}
        threshold = 0.3
        if self.total_frames == 0: return verdict

        if (self.bad_vis_frames / self.total_frames) > threshold:
            verdict["errors"].append("❌ CAMERA ANGLE: Legs are cut off.")
        if (self.bad_track_frames / self.total_frames) > threshold:
            verdict["errors"].append("❌ UNSTABLE: Camera shaking.")
        if (self.bad_light_frames / self.total_frames) > threshold:
            verdict["errors"].append("❌ LIGHTING: Too dark.")
        
        # 2. SMART BLUR CHECK
        # We only reject if >90% of the video is blurry.
        # This catches "Focus Blur" (constant) but ignores "Motion Blur" (temporary).
        if (self.bad_blur_frames / self.total_frames) > 0.90:
            verdict["errors"].append("❌ FOCUS: Camera is out of focus.")
            
        if (self.small_person_frames / self.total_frames) > threshold:
            verdict["errors"].append("❌ DISTANCE: Too far away.")

        if verdict["errors"]: verdict["passed"] = False
        return verdict

@app.post("/upload_video")
async def process_video(file: UploadFile = File(...), mode: str = Form(...)):
    try:
        input_path = os.path.join(TEMP_DIR, file.filename)
        with open(input_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)

        output_filename = f"ai_{file.filename}"
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        cap = cv2.VideoCapture(input_path)
        width, height = int(cap.get(3)), int(cap.get(4))
        
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        if raw_fps is None or raw_fps < 1 or raw_fps > 120:
            fps = 30
        else:
            fps = int(round(raw_fps))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

        rep_count = 0
        session_stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        reps_log = [] 
        
        is_squatting = False
        ai_votes = [] 
        current_feedback = "Analyzing..."
        feedback_color = (255, 255, 255)
        
        current_standing_max = 0
        rep_start_frame = 0
        rep_top_base = 0
        min_rep_knee = 180
        
        quality_gate = QualityGate()
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_idx += 1

            results = model(frame)
            annotated_frame = results[0].plot(boxes=False, probs=False)
            h, w, _ = annotated_frame.shape
            font_scale = h / 950.0
            thick = max(2, int(font_scale * 2.5))

            if not results[0].boxes:
                quality_gate.bad_vis_frames += 1
                quality_gate.total_frames += 1
                out.write(annotated_frame)
                continue

            kps_obj = results[0].keypoints
            box_obj = results[0].boxes[0]
            
            quality_issues = quality_gate.check_frame(frame, kps_obj, box_obj, h, w)
            if quality_issues:
                warning_text = " | ".join(quality_issues)
                cv2.putText(annotated_frame, f"⚠️ {warning_text}", (20, h - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (0, 0, 255), 2)

            if kps_obj is not None and len(kps_obj.xy[0]) >= 17:
                kps = kps_obj.xy[0].cpu().numpy()
                
                if mode == "squat":
                    l_knee = get_angle(kps[11], kps[13], kps[15])
                    r_knee = get_angle(kps[12], kps[14], kps[16])
                    l_hip  = get_angle(kps[5], kps[11], kps[13])
                    r_hip  = get_angle(kps[6], kps[12], kps[14])
                    mid_shoulder = (kps[5] + kps[6]) / 2
                    mid_hip = (kps[11] + kps[12]) / 2
                    lean_angle = get_vertical_angle(mid_shoulder, mid_hip)
                    avg_knee = (l_knee + r_knee) / 2

                    if not is_squatting:
                        current_standing_max = max(current_standing_max, avg_knee)

                    if avg_knee < 165:
                        if not is_squatting:
                            is_squatting = True
                            rep_start_frame = frame_idx
                            rep_top_base = current_standing_max if current_standing_max > 0 else avg_knee
                            min_rep_knee = avg_knee 
                            current_standing_max = 0

                        min_rep_knee = min(min_rep_knee, avg_knee)
                        
                        if squat_classifier:
                            feature_map = {
                                'left_knee_angle': l_knee, 'right_knee_angle': r_knee,
                                'left_hip_angle': l_hip, 'right_hip_angle': r_hip,
                                'torso_lean': lean_angle, 'spine_angle': lean_angle
                            }
                            row = [feature_map.get(col, 0) for col in squat_features]
                            pred = squat_classifier.predict(pd.DataFrame([row], columns=squat_features))[0]
                            
                            if avg_knee < 110:
                                if pred == 1: pred = 0
                                if pred == 3 or pred == 5: pred = 0

                            ai_votes.append(pred)
                            current_feedback = SQUAT_LABELS.get(pred, "")
                            feedback_color = (0, 255, 0) if pred == 0 else (0, 0, 255)

                    elif avg_knee > 168 and is_squatting:
                        is_squatting = False
                        
                        duration_frames = frame_idx - rep_start_frame
                        bottom_angle = min_rep_knee
                        depth_delta = rep_top_base - bottom_angle
                        
                        status = "Unknown"
                        
                        if len(ai_votes) > 2:
                            counts = Counter(ai_votes)
                            winner = counts.most_common(1)[0][0]
                            if winner in session_stats: session_stats[winner] += 1
                            if winner != 1: 
                                rep_count += 1
                                status = "Valid"
                            else:
                                status = "Missed_Shallow"
                        else:
                            status = "Missed_BadData"
                        
                        reps_log.append({
                            "id": len(reps_log) + 1,
                            "depth": depth_delta,
                            "duration_sec": duration_frames / fps,
                            "status": status,
                            "error_code": winner if status == "Valid" or status == "Missed_Shallow" else -1
                        })

                        ai_votes = [] 
                        current_feedback = "Stand Tall"
                        feedback_color = (255, 255, 255)

                elif mode == "pushup":
                    l_elbow = get_angle(kps[5], kps[7], kps[9])
                    if l_elbow < 90: is_squatting = True
                    if l_elbow > 160 and is_squatting:
                        rep_count += 1
                        is_squatting = False
                    current_feedback = "Push Up"

                cv2.rectangle(annotated_frame, (0, 0), (int(w*0.55), int(h*0.15)), (20, 20, 20), -1)
                start_y = int(h*0.05)
                cv2.putText(annotated_frame, f"REPS: {rep_count}", (20, start_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)
                cv2.putText(annotated_frame, f"{current_feedback}", (20, start_y + int(h*0.06)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, thick)
            
            out.write(annotated_frame)

        cap.release()
        
        gate_result = quality_gate.get_verdict()

        # BURN SUMMARY
        summary_bg = np.zeros((height, width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = height / 750.0
        
        def draw_centered(img, text, y, scale, color):
            text_size = cv2.getTextSize(text, font, scale, 2)[0]
            text_x = (width - text_size[0]) // 2
            cv2.putText(img, text, (text_x, y), font, scale, color, 2)

        for _ in range(fps * 5):
            summary_frame = summary_bg.copy()
            y_pos = int(height * 0.15)
            
            if not gate_result["passed"]:
                draw_centered(summary_frame, "⚠️ VIDEO REJECTED ⚠️", y_pos, base_scale*1.5, (0, 0, 255))
                y_pos += int(height * 0.15)
                for error in gate_result["errors"]:
                    draw_centered(summary_frame, error, y_pos, base_scale, (0, 165, 255))
                    y_pos += int(height * 0.08)
            else:
                draw_centered(summary_frame, "WORKOUT REPORT", y_pos, base_scale*1.5, (255, 255, 255))
                y_pos += int(height * 0.13)
                draw_centered(summary_frame, f"VALID REPS: {rep_count}", y_pos, base_scale*1.2, (0, 255, 255))
                y_pos += int(height * 0.12)
                
                draw_centered(summary_frame, f"Perfect Form: {session_stats[0]}", y_pos, base_scale, (0, 255, 0))
                y_pos += int(height * 0.08)
                if session_stats[2] > 0:
                    draw_centered(summary_frame, f"Lean Fwd: {session_stats[2]}", y_pos, base_scale, (0, 165, 255))
                    y_pos += int(height * 0.08)
                if session_stats[3] > 0:
                    draw_centered(summary_frame, f"Knee Cave: {session_stats[3]}", y_pos, base_scale, (0, 165, 255))

            out.write(summary_frame)
        out.release()
        
        # GENERATE FEEDBACK
        feedback_lines = []
        if not gate_result["passed"]:
            feedback_lines.append("❌ Video Rejected: " + ", ".join(gate_result["errors"]))
        else:
            total_attempts = len(reps_log)
            valid_reps = [r for r in reps_log if r['status'] == 'Valid']
            missed_reps = [r for r in reps_log if 'Missed' in r['status']]
            
            if total_attempts > 0:
                valid_pct = (len(valid_reps) / total_attempts) * 100
                missed_pct = (len(missed_reps) / total_attempts) * 100
                
                feedback_lines.append(f"📊 Summary: {len(valid_reps)} Valid / {total_attempts} Total ({valid_pct:.1f}%)")
                
                if valid_pct < 80:
                    feedback_lines.append("⚠️ Accuracy Low: Try to perform the movement more slowly and complete the full cycle.")
                if missed_pct > 20:
                    feedback_lines.append("⚠️ High Miss Rate: Check if you are going deep enough or if the camera angle is cutting off your legs.")

                shallow_count = len([r for r in missed_reps if r['status'] == 'Missed_Shallow'])
                if shallow_count > 0:
                    feedback_lines.append(f"📉 Too Shallow ({shallow_count} reps): Try squatting a little deeper. Keep chest up.")

                if len(valid_reps) > 1:
                    depths = [r['depth'] for r in valid_reps]
                    avg_depth = np.mean(depths)
                    depth_range = max(depths) - min(depths)
                    durations = [r['duration_sec'] for r in valid_reps]
                    avg_tempo = np.mean(durations)

                    if avg_depth < 35:
                        feedback_lines.append(f"📏 Avg Depth {avg_depth:.1f}°: A bit shallow. Aim for >35° change.")
                    else:
                        feedback_lines.append(f"✅ Avg Depth {avg_depth:.1f}°: Good range of motion!")

                    if depth_range > 10:
                        feedback_lines.append(f"⚠️ Inconsistent: Depth varied by {depth_range:.1f}°. Try using a chair/box as a target.")
                    else:
                        feedback_lines.append("✅ Consistent Form: Your reps look very uniform.")

                    if avg_tempo < 0.8:
                        feedback_lines.append(f"⚡ Tempo {avg_tempo:.1f}s: Too fast! Slow down for better control.")
                    elif avg_tempo > 3.0:
                        feedback_lines.append(f"🐢 Tempo {avg_tempo:.1f}s: A bit slow, but good for control.")
                    else:
                        feedback_lines.append(f"✅ Tempo {avg_tempo:.1f}s: Perfect speed.")
            else:
                feedback_lines.append("ℹ️ No complete reps detected.")

        final_feedback_text = "\n".join(feedback_lines)

        return JSONResponse(content={
            "filename": output_filename,
            "rep_count": rep_count,
            "stats": session_stats,
            "passed": gate_result["passed"],
            "analysis_feedback": final_feedback_text 
        })

    except Exception as e:
        print(f"Error: {e}")
        if 'out' in locals(): out.release()
        if 'cap' in locals(): cap.release()
        raise HTTPException(status_code=500, detail="Processing failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)