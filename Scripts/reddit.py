import requests
import os
import subprocess
import time
import cv2

# --- CONFIGURATION ---
TOTAL_GOAL = 300
DEST_DIR = "data/squats"
COACH_WORDS = [
    "depth", "knees", "hips", "back", "heels", "stance", "lower", "drive", "form", "bar",
    "weight", "lean", "feet", "parallel", "ankle", "mobility", "chest", "upright", "butt", "wink", "brace"
]
BOT_BLACKLIST = ["i am a bot", "action was performed automatically", "unsolicited advice", "keep quiet"]
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

def detect_motion_window(video_path):
    """Finds when the person is actually moving to bookmark the action."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    motion_frames = []
    frame_count = 0
    ret, prev_frame = cap.read()
    if not ret: return "00:00 - End"

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        diff = cv2.absdiff(frame, prev_frame)
        if cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)) > 5000:
            motion_frames.append(frame_count)
        prev_frame = frame
        if frame_count > 15000: break
    cap.release()
    if not motion_frames: return "Full Video"
    return f"{time.strftime('%M:%S', time.gmtime(min(motion_frames)/fps))} - {time.strftime('%M:%S', time.gmtime(max(motion_frames)/fps))}"

def get_coaching_feedback(post_url):
    """Fetches high-quality human coaching comments."""
    json_url = post_url.rstrip('/') + ".json"
    feedback = []
    try:
        response = requests.get(json_url, headers=HEADERS)
        data = response.json()
        comments = data[1]['data']['children']
        for c in comments:
            if c['kind'] == 't1':
                body = c['data'].get('body', '')
                low_body = body.lower()
                is_bot = any(b in low_body for b in BOT_BLACKLIST)
                is_long = len(body.split()) > 20
                has_keyword = any(w in low_body for w in COACH_WORDS)
                
                if not is_bot and len(body.split()) > 5:
                    if has_keyword or is_long:
                        clean_text = body.strip().replace('\n', ' ')
                        feedback.append(f"- {clean_text}")
    except: pass
    return feedback

def download_squat_dataset(sub, limit=500):
    """Searches a subreddit and downloads new videos until goal is met."""
    if not os.path.exists(DEST_DIR): os.makedirs(DEST_DIR, exist_ok=True)
    
    # Check count before starting subreddit
    current_count = len([f for f in os.listdir(DEST_DIR) if f.endswith('.mp4')])
    if current_count >= TOTAL_GOAL:
        return True

    url = f"https://www.reddit.com/r/{sub}/search.json?q=squat&restrict_sr=1&limit={limit}&sort=relevance"
    print(f"\n--- Searching r/{sub} | Current: {current_count}/{TOTAL_GOAL} ---")
    
    try:
        response = requests.get(url, headers=HEADERS)
        posts = response.json()['data']['children']
        for post in posts:
            # Check count inside the loop for real-time stopping
            current_count = len([f for f in os.listdir(DEST_DIR) if f.endswith('.mp4')])
            if current_count >= TOTAL_GOAL:
                print(f"!!! GOAL REACHED ({TOTAL_GOAL} videos) !!!")
                return True

            p = post['data']
            if p.get('is_video'):
                post_url = f"https://www.reddit.com{p['permalink']}"
                title = p.get('title', 'No Title')
                clean_title = "".join([c for c in title[:45] if c.isalnum() or c==' ']).strip()
                video_path = os.path.join(DEST_DIR, f"{clean_title}.mp4")
                
                if os.path.exists(video_path): continue 

                print(f"Downloading: {clean_title}")
                try:
                    subprocess.run(['yt-dlp', '-f', 'bestvideo[height<=1080]+bestaudio/best', '-o', video_path, '--merge-output-format', 'mp4', post_url], check=True)
                    
                    action = detect_motion_window(video_path)
                    feedback = get_coaching_feedback(post_url)
                    
                    with open(os.path.join(DEST_DIR, f"{clean_title}_feedback.txt"), "w", encoding="utf-8") as f:
                        f.write(f"TITLE: {title}\nURL: {post_url}\nESTIMATED ACTION: {action}\n\nFEEDBACK:\n")
                        f.write("\n".join(feedback) if feedback else "[No human feedback found]")
                    
                    # 10 second sleep to respect Reddit's servers
                    time.sleep(20)
                except Exception as e:
                    print(f"Download error: {e}")
    except Exception as e:
        print(f"Search error on r/{sub}: {e}")
    return False

if __name__ == "__main__":
    target_subs = [
        "squats", "formcheck", "StartingStrength", "powerlifting", 
        "weightlifting", "crossfit", "Gym", "Lifting", 
        "bodybuilding", "strength_training"
    ]
    
    for s in target_subs:
        finished = download_squat_dataset(s)
        if finished:
            break

    print("\nScript finished.")