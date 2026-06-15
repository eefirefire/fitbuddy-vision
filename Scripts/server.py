import os
import re
import json
import uuid
import shutil
import tempfile

import cv2
from PIL import Image
from google import genai
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Client & Model ────────────────────────────────────────────────────────────
client = genai.Client(api_key="GEMINI_API_KEY_HERE")
ANALYST_MODEL_ID = "models/gemini-3.1-flash-lite-preview"

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_DIR = os.path.join(_HERE, "Official")
RESULTS_DIR  = os.path.join(_HERE, "results")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Expert Biomechanical Analyst specializing in powerlifting squat analysis.
You will be provided with a chronological grid of visual frames spanning an entire training set.

### STRUCTURAL CHAIN-OF-THOUGHT VALIDATION PROTOCOL:
Before producing the final JSON payload, you MUST execute a two-phase internal review:

PHASE 1 — FAULT TRIAGE:
  For each candidate fault you detect, ask:
    (a) Does this fault appear in MORE THAN ONE frame, or is it an isolated single-frame anomaly?
    (b) Is this a true technical failure or a minor balance micro-correction that resolves itself?
  Only escalate a fault to "needs_improvement" if it fails BOTH checks — i.e., it is recurring AND uncorrected.

PHASE 2 — REP SCOPE VERIFICATION:
  Count the exact number of complete squat repetitions visible in the full frame matrix.
  A repetition is only valid if BOTH the descent AND ascent phases are captured across the timeline.
  Do not count partial reps or rack-in/rack-out movements.

After both phases are complete, produce the final JSON payload below. Output ONLY raw JSON — no markdown fences.

OUTPUT SCHEMA (follow exactly):
{
  "exercise": "<string — e.g. barbell_back_squat>",
  "rep_count": <integer>,
  "estimated_load": "<light | moderate | heavy | maximal>",
  "overall_rating": <integer 1-10>,
  "label": "<Perfect Form | Too Shallow | Forward Lean | Knee Cave | Heel Lift | Uneven Balance | Mixed Faults>",
  "good": [
    { "aspect": "<string>", "detail": "<string>" }
  ],
  "needs_improvement": [
    { "aspect": "<string>", "detail": "<string>" }
  ],
  "feedback": "<2-3 sentence plain-English coaching summary>",
  "visual_anchors": { "start_timestamp": "00:00", "end_timestamp": "00:00" }
}
"""

# ── Helpers (ported from few-shot.py) ────────────────────────────────────────

def normalize_token(filename: str) -> str:
    clean = filename.lower()
    clean = re.sub(r'\.(mp4|json|txt)$', '', clean)
    clean = re.sub(r'^(sam3_audited_|sam3_audit_|sam3_accurate_|sam3_redo_|pose_output_)', '', clean)
    clean = clean.replace('_structured_feedback', '').replace('structured_feedback', '')
    clean = clean.replace('_feedback', '').replace('_feeback', '')
    clean = clean.replace('_', ' ')
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def parse_to_dict_safely(text_payload: str) -> dict:
    try:
        clean = text_payload.strip()
        ticks = '```'
        clean = re.sub(rf'^{ticks}(?:json)?', '', clean, flags=re.IGNORECASE)
        clean = re.sub(rf'{ticks}$', '', clean)
        return json.loads(clean.strip())
    except Exception:
        return {"parsing_error": True, "raw_text": text_payload}


def extract_temporal_frame_matrices(video_path: str, max_frames: int = 15) -> list:
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
        frame_rgb    = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(frame_rgb))
        if len(pil_images) >= max_frames:
            break

    cap.release()
    return pil_images


def _load_exemplar_json() -> dict | None:
    """
    Walk the results tree to find a structured_feedback JSON that contains
    a parsed 'zero_shot' or 'rag_augmented' block usable as an exemplar answer.
    Returns the first clean parsed dict found, or None.
    """
    for root, _, files in os.walk(RESULTS_DIR):
        for fname in files:
            if 'structured_feedback' not in fname.lower() or not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(root, fname), encoding='utf-8') as f:
                    blob = json.load(f)

                # The structured_feedback files store raw model text under keys like
                # "zero_shot" or "rag_augmented". Parse whichever is available.
                for key in ("zero_shot", "rag_augmented"):
                    if key in blob:
                        parsed = parse_to_dict_safely(blob[key])
                        if not parsed.get("parsing_error"):
                            return parsed
            except Exception:
                continue
    return None


def _find_exemplar_video() -> str | None:
    """Return the path to one Official video to use as a one-shot visual exemplar."""
    if not os.path.isdir(OFFICIAL_DIR):
        return None
    for fname in os.listdir(OFFICIAL_DIR):
        if fname.lower().endswith('.mp4') and not fname.lower().startswith('temp_'):
            return os.path.join(OFFICIAL_DIR, fname)
    return None


# ── One-Shot + CoT prompt builder ────────────────────────────────────────────

def build_contents_payload(target_frames: list) -> list:
    """
    Constructs:
      [system_prompt] + [exemplar frames] + [exemplar answer] + [target frames] + [task instruction]

    If no exemplar is available, falls back to zero-shot (still CoT-validated).
    """
    contents = [SYSTEM_PROMPT]

    exemplar_json  = _load_exemplar_json()
    exemplar_video = _find_exemplar_video()

    if exemplar_json and exemplar_video:
        ex_frames = extract_temporal_frame_matrices(exemplar_video, max_frames=10)
        if ex_frames:
            contents.extend(ex_frames)
            contents.append(
                "=== ONE-SHOT EXEMPLAR — GROUND-TRUTH ANSWER ===\n"
                + json.dumps(exemplar_json, indent=2)
                + "\n=== END EXEMPLAR ==="
            )

    contents.extend(target_frames)
    contents.append(
        "### CURRENT ASSESSMENT TARGET:\n"
        "Apply the two-phase Chain-of-Thought Validation Protocol above to this frame matrix, "
        "then output ONLY the final raw JSON payload."
    )

    return contents


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="FitBuddy Lift Analyser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze-lift")
async def analyze_lift(
    file: UploadFile = File(...),
    sam3_enabled: str = Form("false"),
):
    # ── 1. Persist upload to a temp file ─────────────────────────────────────
    suffix = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
    tmp_path = os.path.join(tempfile.gettempdir(), f"lift_{uuid.uuid4().hex}{suffix}")

    try:
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

        # ── 2. Extract temporal frame matrix ─────────────────────────────────
        target_frames = extract_temporal_frame_matrices(tmp_path, max_frames=15)
        if not target_frames:
            raise HTTPException(status_code=422, detail="Could not extract frames from the uploaded video.")

        # ── 3. Build one-shot + CoT contents payload ──────────────────────────
        contents = build_contents_payload(target_frames)

        # ── 4. Call Gemini ─────────────────────────────────────────────────────
        use_sam3 = sam3_enabled.lower() in ("true", "1", "yes")

        # When SAM3 is enabled, append a note so the model knows pose keypoints
        # were pre-processed — it should weight joint-angle evidence more heavily.
        if use_sam3:
            contents.append(
                "[SAM3 MODE] This video has been pre-processed with skeleton keypoint overlays. "
                "Trust joint-angle evidence over raw pixel appearance when assessing depth and alignment."
            )

        resp = client.models.generate_content(
            model=ANALYST_MODEL_ID,
            contents=contents,
        )

        # ── 5. Parse and return ────────────────────────────────────────────────
        result = parse_to_dict_safely(resp.text)
        return JSONResponse(content=result)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
