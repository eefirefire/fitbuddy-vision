import os
import cv2
import mediapipe as mp
import numpy as np

# --- INITIALIZATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQUAT_DIR = os.path.join(BASE_DIR, "data", "squats")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Visual styling configurations for a sharp green skeleton overlay
TRACKER_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4)
BONE_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)

def calculate_angle(a, b, c):
    """Calculates the joint angle at a specific vertex (point b)."""
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def run_pose_analysis(video_name):
    video_path = os.path.join(SQUAT_DIR, video_name)
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_name}")
        return

    print(f"🚀 Launching Stabilized BlazePose Heavy Engine for: {video_name}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # This automatically prefixes the output file with POSE_OUTPUT_
    output_path = os.path.join(SQUAT_DIR, f"POSE_OUTPUT_{video_name}")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    
    # 🧠 STABILIZATION FIX: Using Heavy Complexity (2) and high confidence boundaries
    with mp_pose.Pose(
        model_complexity=2,           # Heavy model topology for maximum anatomical lock
        min_detection_confidence=0.65,     # Prevents early initialization onto background objects
        min_tracking_confidence=0.65       # Hard-locks skeleton tracking state to the body
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            
            image_rgb.flags.writeable = True
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Read joints for both sides to stay stable during stance changes
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                    left_angle = calculate_angle(l_hip, l_knee, l_ankle)
                    right_angle = calculate_angle(r_hip, r_knee, r_ankle)
                    
                    # Dynamically prioritize the leg with better visibility metrics
                    l_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
                    r_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
                    active_angle = left_angle if l_vis > r_vis else right_angle
                    
                    cv2.putText(frame, f"Knee Angle: {int(active_angle)} deg", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    if active_angle < 100:
                        cv2.putText(frame, "STATUS: VALID SQUAT DEPTH", 
                                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                                    
                except Exception:
                    pass
                
                # Render the updated, reinforced structural wireframe onto the video frame
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=TRACKER_STYLE,
                    connection_drawing_spec=BONE_STYLE
                )
                
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 20 == 0:
                print(f"  🎬 Processed {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")
                
    cap.release()
    out.release()
    print(f"✨ Successfully exported stabilized video file: {os.path.basename(output_path)}\n")

if __name__ == "__main__":
    # 🚀 SPECIFIC FILENAMES: Explicitly set target arrays using raw directory file names
    SPECIFIC_VIDEOS = [
        "Bulgarian Split Squats  Quads or Glutes.mp4",
        "bulgarian split squat form check.mp4",
        "Squat stance question.mp4"
    ]
    
    print(f"📋 Sequence initialized to process {len(SPECIFIC_VIDEOS)} specific target videos.")
    print("=" * 60)
    
    for vid in SPECIFIC_VIDEOS:
        run_pose_analysis(vid)