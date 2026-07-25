import os
import time
import cv2
import google.generativeai as genai
from google.api_core import exceptions

# --- CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
# Path based on your VS Code sidebar: data/squats
SOURCE_DIR = "data/squats" 

genai.configure(api_key=API_KEY)

def get_duration(path):
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0: return frame_count / fps
    return 0

def process_reddit():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Folder '{SOURCE_DIR}' not found. Please check path.")
        return

    # 1. Get all videos and sort by duration
    all_videos = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.mp4')]
    video_list = []
    
    print(f"Found {len(all_videos)} Reddit videos. Sorting for maximum efficiency...")
    for v in all_videos:
        dur = get_duration(os.path.join(SOURCE_DIR, v))
        video_list.append({'name': v, 'duration': dur})

    video_list.sort(key=lambda x: x['duration'])
    
    model = genai.GenerativeModel(model_name="models/gemini-3.1-flash-lite-preview")

    for item in video_list:
        video_name = item['name']
        video_path = os.path.join(SOURCE_DIR, video_name)
        # Reddit naming convention: video.mp4 -> video_feedback.txt
        txt_path = video_path.replace(".mp4", "_feedback.txt")

        # Skip if already FIXED
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "ACTION_FIXED: Yes" in content:
                    continue

        print(f"\n--- Fixing Reddit Video: {video_name} ({item['duration']:.1f}s) ---")
        
        success = False
        while not success:
            try:
                # Upload to Google
                video_file = genai.upload_file(path=video_path)
                
                while video_file.state.name == "PROCESSING":
                    time.sleep(3)
                    video_file = genai.get_file(video_file.name)

                # Prompt for precise movement analysis
                prompt = (
                    "Watch this Reddit squat video carefully. "
                    "Identify the start of the first rep and the end of the last rep. "
                    "Return ONLY: START: MM:SS, END: MM:SS"
                )
                
                response = model.generate_content([video_file, prompt])
                
                if response.text:
                    # APPEND the fix to the existing text (Title, URL, etc.)
                    with open(txt_path, 'a', encoding='utf-8') as f:
                        f.write(f"\nACTION_FIXED: Yes\nGEMINI_TIMESTAMPS: {response.text.strip()}\n")
                    
                    print(f"FIXED: {response.text.strip()}")
                    
                    # Cleanup
                    genai.delete_file(video_file.name)
                    success = True 
                    
                    # Reddit videos vary; 20s is a safe median cooldown for Lite
                    time.sleep(20) 
                
            except exceptions.ResourceExhausted:
                print("Quota hit (429). Lite is napping. Retrying in 60s...")
                time.sleep(60)
            except Exception as e:
                print(f"Error on {video_name}: {e}. Moving to next.")
                break 

    print("\n" + "="*30)
    print("REDDIT DATASET FIXED!")
    print("="*30)

if __name__ == "__main__":
    process_reddit()