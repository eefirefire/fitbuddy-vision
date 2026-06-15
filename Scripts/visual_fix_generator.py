import os
import json
import cv2
import torch
import time
import numpy as np
from google import genai
from huggingface_hub import hf_hub_download
from sam3.model_builder import build_sam3_video_predictor

# --- 1. PYTORCH VRAM ALLOCATION OVERRIDES ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

original_torch_load = torch.load
def customized_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  
    return original_torch_load(*args, **kwargs)
torch.load = customized_torch_load

# --- 2. CONFIG & INITIALIZATION ---
HF_TOKEN = "HUGGING_FACE_API_KEY"
client = genai.Client(api_key="GEMINI_API_KEY_HERE") 
BRAIN_MODEL = "gemini-2.5-flash"  

REDDIT_DIR = os.path.dirname(os.path.abspath(__file__))
SQUAT_DIR = os.path.normpath(os.path.join(REDDIT_DIR, "data", "squats"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing AI-Auditor SAM 3 Engine on {DEVICE.upper()}...")

checkpoint_path = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", token=HF_TOKEN)
DEVICES = [torch.cuda.current_device()] if DEVICE == "cuda" else []
predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path, gpus_to_use=DEVICES)

if DEVICE == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 🎯 TARGETED REDO PASSTHROUGH DATASET
    TARGET_CONFIGS = [
        {
            "video_name": "Form check for Squat.mp4", 
            "start_second": 11.0, 
            "default_clothing": "maroon shirt and white sweatpants in the foreground"
        }
    ]
    
    print("=" * 75)
    print("🧠 FIT-BUDDY AI: TARGETED PASSTHROUGH REDO PASS")
    print("=" * 75)
    
    for config in TARGET_CONFIGS:
        video = config["video_name"]
        video_path = os.path.normpath(os.path.join(SQUAT_DIR, video))
        
        if not os.path.exists(video_path):
            print(f"❌ Target asset missing from directory: {video_path}")
            continue
            
        cap_in = cv2.VideoCapture(video_path)
        orig_width  = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = cap_in.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0: fps = 30.0
        
        start_frame_idx = int(config["start_second"] * fps)
        
        # --- PHASE 1: HIGH-SPEED 448x448 VIDEO CUT FOR GEMINI NATIVE AUDITING ---
        print(f"\n🎬 Slicing lightweight 5-second analysis clip for Gemini from {video}...")
        gemini_temp_video = os.path.normpath(os.path.join(REDDIT_DIR, f"temp_gemini_eval.mp4"))
        
        gemini_span_frames = min(int(5.0 * fps), total_frames - start_frame_idx)
        video_writer = cv2.VideoWriter(gemini_temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (448, 448))
        
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        for _ in range(gemini_span_frames):
            ret, frame = cap_in.read()
            if not ret: break
            video_writer.write(cv2.resize(frame, (448, 448)))
        video_writer.release()

        form_diagnosis = "PERFECT EXECUTION"
        concept_phrase = f"the legs and {config['default_clothing']} of the person"
        
        if os.path.exists(gemini_temp_video):
            print(f"📤 Uploading optimized preview video to Gemini...")
            uploaded_video = client.files.upload(file=gemini_temp_video)
            
            while "PROCESSING" in str(uploaded_video.state).upper():
                print(".", end="", flush=True)
                time.sleep(1.5)
                uploaded_video = client.files.get(name=uploaded_video.name)
                
            print(f"\n🧠 Prompting Gemini to run full temporal video tracking review...")
            prompt = f"""
            You are an expert biomechanics coach reviewing a video file containing a single heavy squat repetition. 
            Analyze the continuous movement pattern closely to detect any physical form breakdowns over time:

            1. Knee Valgus Cave: Watch the ascent drive carefully. Do the knees buckle or cave inward as the lifter pushes out of the bottom?
            2. Heel Lift / Foot Rolling: Watch the contact with the platform floor. Do the heels peel upward or shift forward at max depth?
            3. Stripper Squat / Good Morning: Do the hips shoot up rapidly while the chest stays pinned down, turning the lift into a lower back extension?
            4. Butt Wink: Does the lower lumbar spine round or tuck underneath severely at the bottom of the hole?

            DIAGNOSTIC JUDGMENT RULEBOOK:
            - Be highly critical. If you notice ANY form failure across the duration of this video, 'fault_found' MUST be true.
            - Write a sharp diagnostic sentence under 7 words for 'coaching_feedback' explaining the error (e.g., "SEVERE KNEE VALGUS COLLAPSE", "HIPS RISING TOO EARLY", "HEELS PEELING OFF FLOOR").
            
            CRITICAL PROMPT REQUIREMENT FOR 'sam3_target_prompt':
            - The text prompt MUST be a simple, short noun phrase targeting a physical object or garment layer (e.g., if knees cave, output exactly "shorts" or "knee sleeves"; if heels lift, output exactly "shoes").
            - DO NOT output relational, positional, or abstract phrases like 'hips of...', 'joints', or 'knees within...'. SAM 3's text encoder will fail to initialize if the target phrase is overly complex.

            Return ONLY a raw, unquoted JSON object matching this schema exactly:
            {{
                "fault_found": true/false,
                "coaching_feedback": "string",
                "sam3_target_prompt": "string"
            }}
            """
            
            try:
                resp = client.models.generate_content(model=BRAIN_MODEL, contents=[prompt, uploaded_video])
                cleaned_text = resp.text.replace('```json', '').replace('```', '').strip()
                data = json.loads(cleaned_text)
                
                form_diagnosis = data.get("coaching_feedback", "PERFECT EXECUTION").upper()
                concept_phrase = data.get("sam3_target_prompt", concept_phrase)
            except Exception as e:
                print(f"⚠️ Video processing failed: {e}. Falling back to default metrics.")
                
            try:
                client.files.delete(name=uploaded_video.name)
            except Exception:
                pass
            if os.path.exists(gemini_temp_video):
                os.remove(gemini_temp_video)
                    
        print(f"📋 Form Diagnosis: '{form_diagnosis}'")
        print(f"🗣️ Active Tracking Cue: \"{concept_phrase}\"")

        # --- PHASE 2: CUT TEMPORARY TIMELINE VIDEO FOR SAM 3 (150 FRAMES) ---
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        print(f"⏱️ Slicing exactly 150 frames for SAM 3 processing channel...")
        
        optimized_temp_video = os.path.normpath(os.path.join(REDDIT_DIR, "low_vram_temp.mp4"))
        track_limit = min(150, total_frames - start_frame_idx)
        
        temp_writer = cv2.VideoWriter(optimized_temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (448, 448))
        
        frames_written = 0
        for _ in range(track_limit):
            ret, frame = cap_in.read()
            if not ret: break
            temp_writer.write(cv2.resize(frame, (448, 448)))
            frames_written += 1
            
        temp_writer.release()
        cap_in.release()

        if frames_written == 0:
            print("❌ Error: Failed to extract timeline slice.")
            if os.path.exists(optimized_temp_video): os.remove(optimized_temp_video)
            continue

        # --- PHASE 3: SAM 3 PROPAGATION LOOP ---
        print("🧬 Initializing tracking session context...")
        response = predictor.handle_request(request=dict(type="start_session", resource_path=optimized_temp_video))
        session_id = response["session_id"]
        
        predictor.handle_request(request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=concept_phrase,
            negative_text="white letters, captions, overlay text, subtitles, watermark, background weights, mirror reflection"
        ))
        
        print("🧬 Propagating stream features down tracking channel timelines...")
        stream_generator = predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session_id))
        
        mask_history = {}
        for resp_data in stream_generator:
            f_idx = resp_data["frame_index"]
            outputs = resp_data["outputs"]
            
            if outputs is not None and "out_binary_masks" in outputs and len(outputs["out_binary_masks"]) > 0:
                masks = outputs["out_binary_masks"]
                mask_history[f_idx] = masks[0].cpu().numpy().squeeze() if isinstance(masks, torch.Tensor) else masks[0].squeeze()
            
            if f_idx % 25 == 0:
                print(f"  🎬 Mask Sync Pipeline: Frame {f_idx}/{frames_written}...")

        # --- PHASE 4: HIGH-RESOLUTION RENDERING PASS ---
        print("🎬 Rendering precision contours and embedding coach feedback diagnostics...")
        clean_out_name = video.replace(' ', '_').replace('__', '_')
        
        # 🛠️ FILENAME FIX: Implemented explicit SAM3_REDO_ prefixing handle pattern
        output_path = os.path.normpath(os.path.join(REDDIT_DIR, f"SAM3_REDO_{clean_out_name}"))
        out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_width, orig_height))
        
        cap_hq = cv2.VideoCapture(video_path)
        cap_hq.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        
        scale_x = orig_width / 448.0
        scale_y = orig_height / 448.0
        
        last_valid_mask = np.zeros((448, 448), dtype=np.uint8)

        for frame_idx in range(frames_written):
            ret, hq_frame = cap_hq.read()
            if not ret: break
            
            if frame_idx in mask_history:
                active_mask = mask_history[frame_idx]
                last_valid_mask = active_mask.copy()
            else:
                active_mask = last_valid_mask

            binary_mask = np.where(active_mask > 0, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            
            scaled_contours = []
            for cnt in contours:
                scaled_cnt = cnt.copy().astype(np.float32)
                scaled_cnt[:, :, 0] *= scale_x
                scaled_cnt[:, :, 1] *= scale_y
                scaled_contours.append(scaled_cnt.astype(np.int32))
            
            hq_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            cv2.drawContours(hq_mask, scaled_contours, -1, 255, thickness=cv2.FILLED)
            
            overlay = hq_frame.copy()
            overlay[hq_mask == 255] = [0, 0, 255]
            hq_frame = cv2.addWeighted(overlay, 0.4, hq_frame, 0.6, 0)
            cv2.drawContours(hq_frame, scaled_contours, -1, [0, 0, 255], 2, cv2.LINE_AA)
                
            cv2.rectangle(hq_frame, (20, 20), (650, 75), (0, 0, 0), -1) 
            cv2.putText(hq_frame, f"AUDIT: {form_diagnosis}", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            out_writer.write(hq_frame)
            
        cap_hq.release()
        out_writer.release()
        
        # --- PHASE 5: VRAM CLEANUP HANDSHAKE ---
        print("🧹 Terminating active tracking context and wiping GPU footprint...")
        try:
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        except Exception as e:
            print(f"⚠️ Session closing warning: {e}")
        
        if os.path.exists(optimized_temp_video): 
            os.remove(optimized_temp_video)
            
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            
        print(f"✨ COMPLETED: Saved audited analysis clip to {os.path.basename(output_path)}\n")
        
    print("🏁 Loop complete! Target asset passthrough redo pass complete.")