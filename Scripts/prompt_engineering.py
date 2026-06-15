import os
import json
import time
import random
import re
from google import genai

#Champion = Biomechanical Analyst


# 1. SETUP
client = genai.Client(api_key="GEMINI_API_KEY_HERE")
MODEL_ID = "gemini-3.1-flash-lite-preview"

# --- 🧠 THE FINAL PROMPT VAULT ---

# This schema ensures the AI provides advice and detailed mechanics for every rep
SHARED_SCHEMA = """
### 🛠️ OUTPUT SCHEMA (JSON ONLY):
{
  "exercise_identity": { "primary_movement": "String", "variation": "String" },
  "rep_count": integer,
  "rep_log": [ 
    { "rep": 1, "score": 1-10, "mechanics": "Detailed anatomical observation" } 
  ],
  "constructive_feedback": {
    "the_win": "One thing the lifter did perfectly.",
    "the_gap": "The biggest technical limiter observed.",
    "the_fix": "A specific drill or setup change."
  },
  "score_justification": "Detailed logic for matching the TRUTH RATING."
}
"""

SHARED_RULES = """
### 🛡️ FIDELITY & INTEGRITY RULES:
1. TRUTH ANCHOR: Your final average score MUST match 'TRUTH RATING' within +/- 0.2.
2. DEDUCTIVE SCORING: Start at 10.0. Subtract 0.5 for minor faults and 1.0 for major faults.
3. ADVICE TONE: Be a supportive, expert peer. Focus on the TOP technical limiter only.
"""

PROMPTS = {
    "BIOMECHANICAL_ANALYST": f"""You are a Lead Biomechanical Engineer. 
Focus: Joint stacking, moment arms, and kinematic chain efficiency.
Goal: Provide a clinical audit that justifies the 'TRUTH RATING' through physics.
{SHARED_RULES}
{SHARED_SCHEMA}""",

    "STRENGTH_COACH": f"""You are a World-Class Strength Coach. 
Focus: Concentric drive, bracing integrity, and 'Technical Grit' under load.
Goal: Validate the performance effort and provide high-level coaching cues.
{SHARED_RULES}
{SHARED_SCHEMA}""",

    "REDDIT_SKEPTIC": f"""You are a Senior Moderator of r/formcheck. 
Focus: Depth verification (Patella Plane) and spotting compensation patterns.
Goal: Provide a strict but fair audit using the 'Benefit of the Doubt' protocol.
{SHARED_RULES}
{SHARED_SCHEMA}"""
}

# --- 🚀 DATA LOGIC ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results")

def get_keywords(name):
    clean = name.lower()
    for word in ["_description", "_evaluation", "_structured_feedback", ".json", ".mp4", ".txt"]:
        clean = clean.replace(word, "")
    return {w for w in re.findall(r'[a-zA-Z0-9]+', os.path.splitext(clean)[0]) if len(w) > 2}

def run_final_showdown(batch_size=10):
    video_pool, desc_pool = [], []
    
    # Index Files
    for root, _, files in os.walk(DATA_PATH):
        for f in files:
            if f.lower().endswith(".mp4"):
                video_pool.append({"path": os.path.join(root, f), "keywords": get_keywords(f)})
    for root, _, files in os.walk(RESULTS_PATH):
        for f in files:
            if f.lower().endswith(".json"):
                desc_pool.append({"path": os.path.join(root, f), "keywords": get_keywords(f)})

    # Match Pairs
    targets = []
    for d in desc_pool:
        match = next((v for v in video_pool if len(d["keywords"].intersection(v["keywords"])) >= 1), None)
        if match:
            targets.append({"video": match["path"], "description": d["path"], "name": os.path.basename(match["path"])})

    if not targets:
        print("🛑 No matches found. Check your data/results folders.")
        return

    selected = random.sample(targets, min(len(targets), batch_size))
    error_tracker = {k: [] for k in PROMPTS.keys()}

    print(f"🔥 STARTING FINAL SHOWDOWN: {len(selected)} VIDEOS\n")

    for i, target in enumerate(selected, 1):
        print(f"🎬 [{i}/{len(selected)}] Testing: {target['name']}")
        
        # Load the "Truth"
        with open(target["description"], 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                truth_rating = data.get("overall_rating", data.get("rating", 7.5))
                reps = data.get("rep_count", "Check video")
            except:
                truth_rating, reps = 7.5, "Check video"

        # Upload Video
        uploaded = client.files.upload(file=target["video"])
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        # Audit with all 3 Prompts
        for p_name, p_text in PROMPTS.items():
            try:
                resp = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[p_text, f"TRUTH RATING: {truth_rating}", f"EXPECTED REPS: {reps}", uploaded]
                )
                ai_json = json.loads(resp.text.replace('```json', '').replace('```', '').strip())
                
                # Calculate the Error (Absolute Distance from Truth)
                ai_avg = sum([r['score'] for r in ai_json['rep_log']]) / len(ai_json['rep_log'])
                diff = abs(ai_avg - truth_rating)
                error_tracker[p_name].append(diff)
                
                print(f"  ✅ {p_name}: {ai_avg:.2f} (Diff: {diff:.2f})")
            except Exception as e:
                print(f"  ❌ {p_name} failed: {e}")

        client.files.delete(name=uploaded.name)
        time.sleep(2)

    # --- 📊 LEADERBOARD ---
    print("\n" + "="*60)
    print("🏆 ACCURACY LEADERBOARD (Lower Error = Better Match)")
    print("="*60)
    for name, errors in error_tracker.items():
        if errors:
            mae = sum(errors) / len(errors)
            print(f"Prompt {name:22} | Avg Error: {mae:.4f} pts")
    print("="*60)

if __name__ == "__main__":
    run_final_showdown(10)