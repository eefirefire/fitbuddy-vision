import os
import json
import time
import re
import random
import cv2
from google import genai
from PIL import Image

# 1. EXPERIMENT CONFIGURATION
client = genai.Client(api_key="GEMINI_API_KEY_HERE")
ANALYST_MODEL_ID = "models/gemini-3.1-flash-lite-preview"
JUDGE_MODEL_ID = "gemini-2.5-flash"  

TRIALS_PER_STRATEGY = 5  

STRATEGY_CONFIGS = {
    "Zero-Shot": {"shots": 0, "reflection": False, "cot": False},   
    "Three-Shot": {"shots": 3, "reflection": False, "cot": False},  
    "CoT-Verification-Shot": {"shots": 2, "reflection": False, "cot": True},
    "Self-Reflection-Pass": {"shots": 0, "reflection": True, "cot": False},
    "Balanced-Few-Shot": {"shots": 2, "reflection": False, "cot": True}
}

BASE_SYSTEM_INSTRUCTION = """You are an Expert Biomechanical Analyst specializing in powerlifting squat analysis.
You will be provided with a chronological grid of visual frames spanning an entire training set.
Analyze the global movement patterns, count the exact repetition scope across the whole video, and output a precise form rating based on our official rubric.
"""

COT_INSTRUCTION_APPEND = """
### 🧠 STRUCTURAL CHAIN-OF-THOUGHT PROTOCOL:
Before building the final JSON payload, you MUST perform a critical validation review. 
Evaluate each deduction item based on the full timeline sequence. If an error is a minor stabilization anomaly, do not penalize.
"""

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

def run_alignment_judge(expert_json, expert_txt, analyst_json):
    judge_prompt = f"""You are an Expert Biomechanical Audit Judge specializing in sports performance validation.
Your job is to compare an Expert Human Ground-Truth dataset against an AI Analyst's generated report.

### 🎯 CRITICAL AUDIT FOCUS AREA:
Evaluate the technical alignment between the AI's output and BOTH human files combined.
Pay strict attention to whether the AI accurately matched the exercise style, total repetition count scope, and specific biomechanical faults (like butt wink, knee valgus, or torso lean) across the entire timeline.

[EXPERT HUMAN STRUCTURED DATA JSON]:
{json.dumps(expert_json, indent=2)}

[EXPERT HUMAN RAW FEEDBACK TEXT]:
{expert_txt}

[AI ANALYST GENERATED REPORT]:
{json.dumps(analyst_json, indent=2)}

Return your evaluation matrix strictly as a raw, unquoted JSON object matching this layout schema exactly:
{{
    "alignment_score": 8.5,
    "verdict_rationale": "Detail the clinical justification regarding their structured data and raw text alignment in under 3 sentences."
}}
"""
    try:
        resp = client.models.generate_content(model=JUDGE_MODEL_ID, contents=[judge_prompt])
        parsed_verdict = parse_to_dict_safely(resp.text)
        return float(parsed_verdict.get("alignment_score", 5.0)), parsed_verdict.get("verdict_rationale", "Judge rationale unparseable.")
    except Exception:
        return 5.0, "Failed to execute alignment verification pass."

def run_reflection_pass(raw_ai_output):
    reflection_prompt = f"""You are an Expert Biomechanical Verification Auditor. 
Review your draft squat assessment carefully to eliminate false positives or alignment anomalies.
Ensure your parsed repetition count matches the full frame matrix timeline scope.

[DRAFT ASSESSMENT]:
{raw_ai_output}

Return a revised, structurally pristine JSON block following the identical schema rules. Output ONLY JSON."""
    try:
        resp = client.models.generate_content(model=ANALYST_MODEL_ID, contents=[reflection_prompt])
        return resp.text.strip()
    except Exception:
        return raw_ai_output

def run_experiment_suite():
    REDDIT_DIR = os.getcwd()
    project_workspace_root = REDDIT_DIR
    while os.path.basename(project_workspace_root) != "AI_Builder_Fitness" and os.path.dirname(project_workspace_root) != project_workspace_root:
        project_workspace_root = os.path.dirname(project_workspace_root)
        
    print(f"🔍 Initializing Workspace Anchor Point: {project_workspace_root}\n")

    video_list = []
    json_list = []
    txt_list = []

    for root, _, files in os.walk(project_workspace_root):
        root_normalized = root.lower().replace('\\', '/')
        path_segments = root_normalized.split('/')
        
        for f in files:
            f_lower = f.lower()
            full_path = os.path.join(root, f)
            token = normalize_token(f)
            
            # 🎯 HARD-LOCKED STRICT ROUTING DIRECTORY MATCHING
            if f_lower.endswith(".mp4") and "official" in path_segments and not f_lower.startswith("temp_"):
                if path_segments[-1] == "official" or (len(path_segments) > 1 and path_segments[-2] == "official"):
                    video_list.append({"path": full_path, "name": f, "token": token})
                    
            elif f_lower.endswith(".json") and "results" in path_segments and "structured_feedback" in f_lower:
                json_list.append({"path": full_path, "name": f, "token": token})
                
            elif f_lower.endswith(".txt") and "data" in path_segments and "squats" in path_segments:
                if "_feedback" in f_lower or "_feeback" in f_lower:
                    txt_list.append({"path": full_path, "name": f, "token": token})

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

    print(f"📦 Verification Handshake Complete: Paired {len(all_matched_profiles)} multi-source form profiles.")
    if len(all_matched_profiles) == 0:
        print("❌ Error: No profiles matched. Check directory tokens.")
        return

    performance_ledger = {}

    for strategy_name, config in STRATEGY_CONFIGS.items():
        print(f"\n==================================================")
        print(f"🚀 RUNNING ARCHITECTURE: {strategy_name}")
        print(f"==================================================")
        
        alignment_scores_accumulated = []
        successful_runs = 0

        for trial in range(1, TRIALS_PER_STRATEGY + 1):
            print(f"⏱️ [Trial {trial}/{TRIALS_PER_STRATEGY}]")
            
            sample_pool = list(all_matched_profiles)
            test_profile = random.choice(sample_pool)
            sample_pool.remove(test_profile)
            
            try:
                with open(test_profile["json_path"], 'r', encoding='utf-8') as f:
                    expert_json_data = json.load(f)
                with open(test_profile["txt_path"], 'r', encoding='utf-8') as f:
                    expert_txt_data = f.read()
            except Exception:
                print("   ❌ Error reading target profiles.")
                continue

            contents_payload = [BASE_SYSTEM_INSTRUCTION]
            if config["cot"]:
                contents_payload[0] += COT_INSTRUCTION_APPEND
                
            # Interleave Exemplar Training Elements
            available_shots = min(config["shots"], len(sample_pool))
            if available_shots > 0:
                for idx, ex_p in enumerate(random.sample(sample_pool, available_shots)):
                    try:
                        ex_frames = extract_temporal_frame_matrices(ex_p["video_path"])
                        with open(ex_p["json_path"], 'r', encoding='utf-8') as f:
                            ex_json_content = json.dumps(json.load(f), indent=2)
                        with open(ex_p["txt_path"], 'r', encoding='utf-8') as f:
                            ex_txt_content = f.read()
                            
                        contents_payload.extend(ex_frames)
                        contents_payload.append(
                            f"=== EXEMPLAR {idx+1} ANSWER METRICS ===\n"
                            f"Ground-Truth JSON:\n{ex_json_content}\n"
                            f"Text Feedback:\n{ex_txt_content}\n"
                        )
                    except Exception:
                        pass

            print(f"   📊 Extracting Full-Timeline Visual Frame Matrix: {test_profile['video_name']}...")
            target_frames = extract_temporal_frame_matrices(test_profile["video_path"])
            
            if not target_frames:
                print("   ❌ Frame vector extraction failure.")
                continue
                
            contents_payload.extend(target_frames)
            contents_payload.append("### 🎯 CURRENT RUN ASSESSMENT TARGET:\nEvaluate this comprehensive frame matrix timeline sequence blindly. Output raw JSON.")

            print("   🧠 Running multi-modal analysis...")
            try:
                resp = client.models.generate_content(model=ANALYST_MODEL_ID, contents=contents_payload)
                raw_analyst_text = resp.text
                
                if config["reflection"]:
                    raw_analyst_text = run_reflection_pass(raw_analyst_text)
                
                analyst_parsed_dict = parse_to_dict_safely(raw_analyst_text)
                
                print("   ⚖️ Initializing LLM Alignment Judge Review (Targeting JSON & TXT sync)...")
                score, reasoning = run_alignment_judge(expert_json_data, expert_txt_data, analyst_parsed_dict)
                
                alignment_scores_accumulated.append(score)
                successful_runs += 1
                
                print(f"   🎯 Judge Alignment Score: {score}/10.0")
                print(f"   📝 Judge Rationale: \"{reasoning}\"\n")
                
            except Exception as e:
                print(f"   ❌ Network Exception: {e}")

        if successful_runs > 0:
            avg_alignment = sum(alignment_scores_accumulated) / successful_runs
            performance_ledger[strategy_name] = f"{avg_alignment:.2f} / 10.0 Average Alignment Match"
        else:
            performance_ledger[strategy_name] = "Inconclusive Data Profile"

    print(f"\n🏁 ==================================================")
    print(f"📊 AUTOMATED MULTI-SOURCE ALIGNMENT REPORT")
    print(f"==================================================")
    print(f"{'Prompt Strategy Layout':<25} | {'Semantic Agreement Score (Higher is Better)':<40}")
    print("-" * 75)
    for strategy, score_summary in performance_ledger.items():
        print(f"{strategy:<25} | {score_summary:<40}")
    print(f"==================================================\n")

if __name__ == "__main__":
    run_experiment_suite()