"""
Fixed runner for the prompt strategy benchmark.
Patches two issues in experiment.py:
  1. Video lookup required an "official/" folder that doesn't exist  videos live in
     data/reddit/data/squats/*.mp4 alongside their *_feedback.txt files.
  2. API key was a placeholder  read from squat-frontend/.env instead.
Everything else (strategies, judge, reflection, scoring) is identical to experiment.py.
"""
import os, json, time, re, random

# The google-genai SDK uses httpx internally. Patch it before import so all
# Client instances skip certificate verification (user-authorized workaround
# for Windows proxy/firewall SSL interception).
import httpx
_OrigClient  = httpx.Client
_OrigAClient = httpx.AsyncClient
class _NV(httpx.Client):
    def __init__(self, *a, **kw): kw['verify'] = False; super().__init__(*a, **kw)
class _NVA(httpx.AsyncClient):
    def __init__(self, *a, **kw): kw['verify'] = False; super().__init__(*a, **kw)
httpx.Client      = _NV
httpx.AsyncClient = _NVA

import cv2
from google import genai
from PIL import Image

#  Path anchors 
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(SCRIPT_DIR)
SQUATS_DIR   = os.path.join(ROOT, "data", "reddit", "data", "squats")
RESULTS_DIR  = os.path.join(ROOT, "data", "reddit", "results")

#  API key from .env 
ENV_PATH = os.path.join(ROOT, "squat-frontend", ".env")
api_key = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith("VITE_GEMINI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
if not api_key:
    raise RuntimeError(f"Could not read VITE_GEMINI_API_KEY from {ENV_PATH}")

client           = genai.Client(api_key=api_key)
ANALYST_MODEL_ID = "models/gemini-3.1-flash-lite-preview"
JUDGE_MODEL_ID   = "gemini-2.5-flash"
TRIALS_PER_STRATEGY = 5

STRATEGY_CONFIGS = {
    "Zero-Shot":            {"shots": 0, "reflection": False, "cot": False},
    "Three-Shot":           {"shots": 3, "reflection": False, "cot": False},
    "CoT-Verification-Shot":{"shots": 2, "reflection": False, "cot": True},
    "Self-Reflection-Pass": {"shots": 0, "reflection": True,  "cot": False},
    "Balanced-Few-Shot":    {"shots": 2, "reflection": False,  "cot": True},
}

BASE_SYSTEM_INSTRUCTION = """You are an Expert Biomechanical Analyst specializing in powerlifting squat analysis.
You will be provided with a chronological grid of visual frames spanning an entire training set.
Analyze the global movement patterns, count the exact repetition scope across the whole video, and output a precise form rating based on our official rubric.
"""

COT_INSTRUCTION_APPEND = """
### > STRUCTURAL CHAIN-OF-THOUGHT PROTOCOL:
Before building the final JSON payload, you MUST perform a critical validation review.
Evaluate each deduction item based on the full timeline sequence. If an error is a minor stabilization anomaly, do not penalize.
"""

#  Helpers 
def normalize(name):
    n = os.path.splitext(name)[0].lower()
    for s in ("_structured_feedback", "structured_feedback", "_feedback", "_feeback"):
        n = n.replace(s, "")
    return re.sub(r"[^a-z0-9]", "", n)

def parse_json_safely(text):
    try:
        clean = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
        clean = re.sub(r"```$", "", clean).strip()
        return json.loads(clean)
    except Exception:
        return {"parsing_error": True, "raw_text": text}

def extract_frames(video_path, max_frames=15):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return []
    interval = max(1, total // max_frames)
    frames = []
    for i in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        if len(frames) >= max_frames: break
    cap.release()
    return frames

def judge(expert_json, expert_txt, analyst_json):
    prompt = f"""You are an Expert Biomechanical Audit Judge.
Compare the AI Analyst report against BOTH human ground-truth files.
Focus on: exercise style match, repetition count accuracy, specific biomechanical fault identification.

[EXPERT JSON]: {json.dumps(expert_json, indent=2)}
[EXPERT TEXT]: {expert_txt}
[AI REPORT]:   {json.dumps(analyst_json, indent=2)}

Return ONLY this JSON:
{{"alignment_score": 8.5, "verdict_rationale": "..."}}"""
    try:
        resp = client.models.generate_content(model=JUDGE_MODEL_ID, contents=[prompt])
        parsed = parse_json_safely(resp.text)
        return float(parsed.get("alignment_score", 5.0)), parsed.get("verdict_rationale", "")
    except Exception as e:
        return 5.0, f"Judge failed: {e}"

def reflect(raw):
    prompt = f"""You are an Expert Biomechanical Verification Auditor.
Review your draft squat assessment. Eliminate false positives. Ensure rep count matches the full timeline.

[DRAFT]: {raw}

Return ONLY revised JSON."""
    try:
        resp = client.models.generate_content(model=ANALYST_MODEL_ID, contents=[prompt])
        return resp.text.strip()
    except Exception:
        return raw

#  Build matched profiles 
def build_profiles():
    # Index structured_feedback JSONs by normalised name
    json_index = {}
    for root, _, files in os.walk(RESULTS_DIR):
        for f in files:
            if f.lower().endswith(".json") and "structured_feedback" in f.lower():
                json_index[normalize(f)] = os.path.join(root, f)

    profiles = []
    for f in os.listdir(SQUATS_DIR):
        if not f.lower().endswith(".mp4"):
            continue
        base  = os.path.splitext(f)[0]
        token = normalize(base)
        mp4   = os.path.join(SQUATS_DIR, f)

        # txt: same folder, same base name + _feedback.txt
        txt_path = os.path.join(SQUATS_DIR, base + "_feedback.txt")
        if not os.path.exists(txt_path):
            txt_path = os.path.join(SQUATS_DIR, base + "_feeback.txt")
        if not os.path.exists(txt_path):
            continue

        # json: fuzzy match in results
        best_key  = None
        best_score = 0
        for key in json_index:
            if token in key or key in token:
                s = min(len(token), len(key))
                if s > best_score:
                    best_score = s; best_key = key
        if best_key is None:
            continue

        profiles.append({
            "name":     base,
            "video":    mp4,
            "txt":      txt_path,
            "json":     json_index[best_key],
        })
    return profiles

#  Main experiment 
def main():
    profiles = build_profiles()
    print(f"[OK] Matched {len(profiles)} video/json/txt triples\n")
    if not profiles:
        print("[FAIL] No profiles matched. Check directory structure."); return

    ledger = {}

    for strategy, cfg in STRATEGY_CONFIGS.items():
        print(f"\n{'='*52}")
        print(f"RUNNING  {strategy}")
        print(f"{'='*52}")

        scores = []
        for trial in range(1, TRIALS_PER_STRATEGY + 1):
            print(f"  Trial {trial}/{TRIALS_PER_STRATEGY}", end=" ", flush=True)

            pool = list(profiles)
            target = random.choice(pool)
            pool.remove(target)

            try:
                expert_json = json.load(open(target["json"], encoding="utf-8"))
                expert_txt  = open(target["txt"],  encoding="utf-8").read()
            except Exception as e:
                print(f"[FAIL] read error: {e}"); continue

            sys_prompt = BASE_SYSTEM_INSTRUCTION + (COT_INSTRUCTION_APPEND if cfg["cot"] else "")
            contents   = [sys_prompt]

            # Shot exemplars
            for ex in random.sample(pool, min(cfg["shots"], len(pool))):
                try:
                    frames = extract_frames(ex["video"])
                    ex_j   = json.dumps(json.load(open(ex["json"], encoding="utf-8")), indent=2)
                    ex_t   = open(ex["txt"], encoding="utf-8").read()
                    contents.extend(frames)
                    contents.append(f"=== EXEMPLAR ANSWER ===\nJSON:\n{ex_j}\nText:\n{ex_t}\n")
                except Exception:
                    pass

            target_frames = extract_frames(target["video"])
            if not target_frames:
                print("[FAIL] no frames"); continue

            contents.extend(target_frames)
            contents.append("### CURRENT TARGET: Evaluate this frame sequence. Output raw JSON only.")

            try:
                resp = client.models.generate_content(model=ANALYST_MODEL_ID, contents=contents)
                raw  = resp.text
                if cfg["reflection"]:
                    raw = reflect(raw)
                parsed = parse_json_safely(raw)

                score, rationale = judge(expert_json, expert_txt, parsed)
                scores.append(score)
                print(f" {score:.1f}/10  \"{rationale[:80]}...\"")
                time.sleep(1)   # gentle rate-limit

            except Exception as e:
                print(f"[FAIL] {e}")

        if scores:
            avg = sum(scores) / len(scores)
            ledger[strategy] = (avg, scores)
            print(f"  > Average: {avg:.2f}/10  (trials: {scores})")
        else:
            ledger[strategy] = (None, [])

    #  Final report 
    print(f"\n{'='*60}")
    print(f"  PROMPT STRATEGY BENCHMARK  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Strategy':<26} | {'Avg Score':>10} | {'Trials'}")
    print("-" * 60)
    ranked = sorted(ledger.items(), key=lambda x: x[1][0] or 0, reverse=True)
    for name, (avg, trials) in ranked:
        avg_str = f"{avg:.2f}/10" if avg is not None else "N/A"
        print(f"{name:<26} | {avg_str:>10} | {trials}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
