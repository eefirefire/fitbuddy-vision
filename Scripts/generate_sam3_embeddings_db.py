import os
import cv2
import json
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from sam3.model_builder import build_sam3_video_model

# --- 1. PYTORCH SECURITY BYPASS ---
original_torch_load = torch.load
def customized_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  
    return original_torch_load(*args, **kwargs)
torch.load = customized_torch_load

# --- 2. CONFIG & INITIALIZATION ---
HF_TOKEN = "YOUR_HF_READ_TOKEN_HERE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQUAT_DIR = os.path.join(BASE_DIR, "data", "squats")
DB_PATH = os.path.join(BASE_DIR, "sam3_vector_db.json")

# Hand-pick your 2-3 perfect form fault example videos here!
REFERENCE_VIDEOS = [
    "Knees hurt after squats what am I doing wron.mp4",
    "Recently started heellifted squats and.mp4"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing SAM 3 Feature Extractor on {DEVICE.upper()}...")

# Safely download and grab the clean local cache path string
checkpoint_path = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", token=HF_TOKEN)

# Load the base model natively into your GPU memory
model = build_sam3_video_model(checkpoint_path=checkpoint_path).to(DEVICE)
model.eval()

def extract_native_features(video_name, num_samples=5):
    video_path = os.path.join(SQUAT_DIR, video_name)
    if not os.path.exists(video_path):
        print(f"❌ Could not find video file: {video_name}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
        
    # Pick evenly spaced frames across the clip to capture the entire movement shape
    frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    sampled_vectors = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
            
        # Resize to standard input shape for the SAM 3 visual transformer block
        img_resized = cv2.resize(frame, (1024, 1024))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        img_tensor = (img_tensor / 255.0 - 0.5) / 0.5 # Tensor normalization map
        
        with torch.no_grad():
            # Hook right into the model's raw visual layer backbone
            image_embeddings = model.image_encoder(img_tensor)
            
            if isinstance(image_embeddings, dict):
                feature_vector = image_embeddings["vision_features"]
            elif isinstance(image_embeddings, tuple):
                feature_vector = image_embeddings[0]
            else:
                feature_vector = image_embeddings
                
            # Average out spatial arrays to reduce dimension footprint down to 1D
            flat_vector = torch.mean(feature_vector, dim=[2, 3]).cpu().numpy().squeeze()
            sampled_vectors.append(flat_vector)
            
    cap.release()
    
    if len(sampled_vectors) == 0: return None
    # Compile the final coordinate match map
    return np.mean(sampled_vectors, axis=0).tolist()

def main():
    sam3_db = {}
    print(f"📂 Processing {len(REFERENCE_VIDEOS)} selected reference videos...")
    
    for vid in REFERENCE_VIDEOS:
        print(f"🧬 Extracting SAM 3 encoder latent tensors from: {vid}")
        vector = extract_native_features(vid)
        
        if vector is not None:
            sam3_db[vid] = {
                "video_name": vid,
                "sam3_embedding": vector
            }
            
    # Write the compiled vector matrix map cleanly to disk space
    with open(DB_PATH, "w") as f:
        json.dump(sam3_db, f, indent=4)
        
    print(f"\n🚀 SUCCESS! Local SAM 3 vector database created: {DB_PATH}")

if __name__ == "__main__":
    main()