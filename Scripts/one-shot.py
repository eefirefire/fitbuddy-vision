import os
import json
import re
import random
import cv2
from google import genai
from PIL import Image

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE"))
ANALYST_MODEL_ID = "models/gemini-3.1-flash-lite-preview"

BASE_SYSTEM_INSTRUCTION = """You are an Expert Biomechanical Analyst specializing in powerlifting squat analysis.
You will be provided with a chronological grid of visual frames spanning an entire training set.
Analyze the global movement patterns, count the exact repetition scope across the whole video, and output a precise form rating based on our official rubric.
"""

# ==========================================
# 2. FILE PARSING & PIPELINE UTILITIES
# ==========================================
def normalize_token(filename):
    clean = filename.lower()
    clean = re.sub(r'\.(mp4|json|txt)$', '', clean)
    clean = re.sub(r'^(sam3_audited_|sam3_audit_|sam3_accurate_|sam3_redo_|pose_output_)', '', clean)
    clean = clean.replace('_structured_feedback', '').replace('structured_feedback', '')
    clean = clean.replace('_feedback', '').replace('_feeback', '')
    clean = clean.replace('_', ' ')
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def parse_to_dict_safely(text_payload):
    try:
        clean_txt = text_payload.strip()
        ticks = '`' + '`' + '`'
        clean_txt = re.sub(rf'^{ticks}(?:json)?', '', clean_txt, flags=re.IGNORECASE)
        clean_txt = re.sub(rf'{ticks}$', '', clean_txt)
        return json.loads(clean_txt.strip())
    except Exception:
        return {"parsing_error": True, "raw_text": text_payload}

def extract_temporal_frame_matrices(video_path, max_frames=15):
    """Samples frames evenly across the entire duration of the video to match human expert scope."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
        
    interval = max(1, total_frames // max_frames)
    pil_images = []
    
    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(frame_rgb))
        if len(pil_images) >= max_frames:
            break
            
    cap.release()
    return pil_images

# ==========================================
# 3. CORE ONE-SHOT TEST EXECUTION
# ==========================================
def run_single_one_shot_test():
    REDDIT_DIR = os.getcwd()
    project_workspace_root = REDDIT_DIR
    while os.path.basename(project_workspace_root) != "AI_Builder_Fitness" and os.path.dirname(project_workspace_root) != project_workspace_root:
        project_workspace_root = os.path.dirname(project_workspace_root)
        
    print(f"🔍 Anchoring Workspace: {project_workspace_root}")

    video_list = []
    json_list = []
    txt_list = []

    # Crawl target directories using your strict path logic
    for root, _, files in os.walk(project_workspace_root):
        root_normalized = root.lower().replace('\\', '/')
        path_segments = root_normalized.split('/')
        
        for f in files:
            f_lower = f.lower()
            full_path = os.path.join(root, f)
            token = normalize_token(f)
            
            if f_lower.endswith(".mp4") and "official" in path_segments and not f_lower.startswith("temp_"):
                if path_segments[-1] == "official" or (len(path_segments) > 1 and path_segments[-2] == "official"):
                    video_list.append({"path": full_path, "name": f, "token": token})
            elif f_lower.endswith(".json") and "results" in path_segments and "structured_feedback" in f_lower:
                json_list.append({"path": full_path, "name": f, "token": token})
            elif f_lower.endswith(".txt") and "data" in path_segments and "squats" in path_segments:
                if "_feedback" in f_lower or "_feeback" in f_lower:
                    txt_list.append({"path": full_path, "name": f, "token": token})

    # Handshake matching logic to assemble data profiles
    all_matched_profiles = []
    for v in video_list:
        best_j = None
        max_j_score = 0
        for j in json_list:
            if v["token"] in j["token"] or j["token"] in v["token"]:
                score = min(len(v["token"]), len(j["token"]))
                if score > max_j_score:
                    max_j_score = score
                    best_j = j
                    
        best_t = None
        max_t_score = 0
        for t in txt_list:
            if v["token"] in t["token"] or t["token"] in v["token"]:
                score = min(len(v["token"]), len(t["token"]))
                if score > max_t_score:
                    max_t_score = score
                    best_t = t

        if best_j and best_t:
            all_matched_profiles.append({
                "video_name": v["name"],
                "video_path": v["path"],
                "json_path": best_j["path"],
                "txt_path": best_t["path"]
            })

    print(f"📦 Dataset Handshake Complete: Paired {len(all_matched_profiles)} multi-source form profiles.")
    if len(all_matched_profiles) < 2:
        print("❌ Error: Need at least 2 matched profiles to assign an Exemplar and a Test Target.")
        return

    # Extract two profiles: one context exemplar, one blind target
    sample_pool = list(all_matched_profiles)
    exemplar_profile = random.choice(sample_pool)
    sample_pool.remove(exemplar_profile)
    test_profile = random.choice(sample_pool)

    print(f"\n💡 [ONE-SHOT EXEMPLAR SELECTION]: {exemplar_profile['video_name']}")
    print(f"🎯 [LIVE INFERENCE TEST TARGET]: {test_profile['video_name']}\n")

    contents_payload = [BASE_SYSTEM_INSTRUCTION]

    # Process and append the real data context profile as the single visual/text exemplar
    try:
        ex_frames = extract_temporal_frame_matrices(exemplar_profile["video_path"])
        with open(exemplar_profile["json_path"], 'r', encoding='utf-8') as f:
            ex_json_content = json.dumps(json.load(f), indent=2)
        with open(exemplar_profile["txt_path"], 'r', encoding='utf-8') as f:
            ex_txt_content = f.read()
            
        contents_payload.extend(ex_frames)
        contents_payload.append(
            f"=== EXEMPLAR 1 ANSWER METRICS ===\n"
            f"Ground-Truth JSON:\n{ex_json_content}\n"
            f"Text Feedback:\n{ex_txt_content}\n"
            f"Use the format layout structure from this verified exemplar to compile your answer.\n"
        )
    except Exception as e:
        print(f"❌ Failed to build context mapping from exemplar pair: {e}")
        return

    # Extract target test frames
    print("📸 Slicing live target frames into chronological matrix...")
    target_frames = extract_temporal_frame_matrices(test_profile["video_path"])
    if not target_frames:
        print("❌ Failed to parse test target frames.")
        return
        
    contents_payload.extend(target_frames)
    contents_payload.append("### 🎯 CURRENT RUN ASSESSMENT TARGET:\nEvaluate this comprehensive frame matrix timeline sequence blindly. Output raw JSON mirroring the exemplar layout format.")

    # Execute direct pass
    print(f"🤖 Launching One-Shot inference request via {ANALYST_MODEL_ID}...")
    try:
        resp = client.models.generate_content(
            model=ANALYST_MODEL_ID,
            contents=contents_payload,
            config={"response_mime_type": "application/json", "temperature": 0.1}
        )
        
        parsed_ai_report = parse_to_dict_safely(resp.text)
        print("\n📊 --- AI MODEL ONE-SHOT RESPONSE PAYLOAD ---")
        print(json.dumps(parsed_ai_report, indent=4))
        
    except Exception as network_err:
        print(f"❌ Generation Execution Interrupted: {network_err}")

if __name__ == "__main__":
    run_single_one_shot_test()