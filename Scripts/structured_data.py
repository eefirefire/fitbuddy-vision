import os
import json
import time
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions

# 1. SETUP
API_KEY = "GEMINI_API_KEY_HERE"
genai.configure(api_key=API_KEY)

# Using the stable Lite model for analysis
MODEL_NAME = "models/gemini-3.1-flash-lite-preview"
model = genai.GenerativeModel(model_name=MODEL_NAME)

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "data", "squats")
CSV_PATH = os.path.join(BASE_DIR, "summarized_feedback.csv")

MASTER_PROMPT = """
You are an expert Biomechanical AI Coach. Perform a high-precision analysis of the squat video and generate a structured JSON.

### 🔍 MANDATORY INSPECTION:
1. FOOT & HEEL STABILITY: Watch for heel lift.
2. KNEE PATH (LATERAL): Watch for medial collapse (valgus).
3. DEPTH (HIP CREASE): Check if it passes below the top of the patella.
4. LUMBAR & PELVIC TILT: Watch for 'butt wink'.
5. UPPER BODY & BAR PATH: Check for chest fall.
6. SAFETY: Evaluate safety pins and bailing.

### 🛠️ OUTPUT SCHEMA (STRICT JSON ONLY):
{
  "exercise": "barbell_back_squat",
  "rep_count": 1,
  "estimated_load": "heavy",
  "overall_rating": 8,
  "good": [{"aspect": "string", "detail": "string"}],
  "needs_improvement": [{"aspect": "string", "detail": "string"}],
  "visual_anchors": {"start_timestamp": "MM:SS", "end_timestamp": "MM:SS"}
}
"""

# 2. HARDENED UPLOAD LOGIC (Fixes SSLEOFError)
def upload_and_wait(video_path):
    video_name = os.path.basename(video_path)
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"📤 Uploading {video_name} (Attempt {attempt + 1}/{max_retries})...")
            video_file = genai.upload_file(path=video_path)
            
            while video_file.state.name == "PROCESSING":
                time.sleep(3)
                video_file = genai.get_file(video_file.name)
            
            return video_file
            
        except Exception as e:
            # Catching SSL, Connection, and EOF errors specifically
            if "SSL" in str(e) or "EOF" in str(e) or "connection" in str(e).lower():
                print(f"🔄 Connection drop on {video_name}. Resting 5s before retry...")
                time.sleep(5) 
            else:
                # If it's a different error (like File Not Found), raise it immediately
                print(f"❌ Unexpected Error: {e}")
                raise e
                
    raise ConnectionError(f"❌ Failed to upload {video_name} after {max_retries} attempts.")

# 3. KEYWORD-BASED RAG
def find_relevant_context_lite(query_text, df):
    keywords = ["fail", "heavy", "light", "depth", "knee", "heel", "valgus", "wink"]
    query_text = query_text.lower()
    scores = []
    for _, row in df.iterrows():
        desc = str(row['video_description']).lower()
        match_score = sum(1 for word in keywords if word in query_text and word in desc)
        scores.append(match_score)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i])[-2:]
    results = df.iloc[top_indices]
    
    context = "\nMATCHED REFERENCE CASES:\n"
    for _, row in results.iterrows():
        context += f"- Past Video: {row['video_description']}\n  Expert Analysis: {row['structured_feedback']}\n"
    return context

# 4. CORE TASK ENGINE
def run_research_task(video_path, df_library):
    video_name = os.path.basename(video_path).split('.')[0]
    output_dir = os.path.join(BASE_DIR, "results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    video_file = upload_and_wait(video_path)

    # Step A: Description
    desc = model.generate_content([video_file, "Describe this squat biomechanically."]).text
    with open(os.path.join(output_dir, f"{video_name}_description.json"), "w") as f:
        json.dump({"description": desc}, f, indent=2)

    # Step B: Zero-Shot (Exp A)
    exp_a_res = model.generate_content([video_file, MASTER_PROMPT]).text

    # Step C: RAG-Augmented (Exp B)
    context = find_relevant_context_lite(desc, df_library)
    rag_prompt = f"{MASTER_PROMPT}\n\n{context}\n\nRefine analysis using matched references."
    exp_b_res = model.generate_content([video_file, rag_prompt]).text
    
    # Save Final JSON
    final_output = {"zero_shot": exp_a_res, "rag_augmented": exp_b_res}
    with open(os.path.join(output_dir, f"{video_name}_structured_feedback.json"), "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"✅ Finished Processing: {video_name}")
    genai.delete_file(video_file.name)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print("🚀 Library CSV not found. Building initial library...")
        library_data = []
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
        
        for filename in video_files:
            v_path = os.path.join(VIDEO_DIR, filename)
            try:
                v_file = upload_and_wait(v_path)
                desc = model.generate_content([v_file, "Describe this squat."]).text
                struct = model.generate_content([v_file, MASTER_PROMPT]).text
                library_data.append({"video_description": desc, "structured_feedback": struct})
                genai.delete_file(v_file.name)
                # Small delay between items to avoid spamming the SSL port
                time.sleep(10) 
            except Exception as e:
                print(f"⚠️ Skipping {filename} due to repeated errors: {e}")
        
        pd.DataFrame(library_data).to_csv(CSV_PATH, index=False)
        print("📂 Library CSV created successfully.")

    df_library = pd.read_csv(CSV_PATH)
    for filename in os.listdir(VIDEO_DIR):
        if filename.endswith(".mp4"):
            run_research_task(os.path.join(VIDEO_DIR, filename), df_library)