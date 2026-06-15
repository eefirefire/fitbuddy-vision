import os
import shutil
import re
from collections import Counter

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def clean(text):
    return re.sub(r'[^a-zA-Z0-9]', '', str(text).lower())

def get_biomechanical_subgroup(folder_name):
    """Determines the Tier 2 group based on specific squat variations."""
    name = folder_name.lower()
    if 'front' in name: return "Front_Squats"
    if 'pause' in name or 'paused' in name: return "Pause_Squats"
    if 'low bar' in name or 'lowbar' in name: return "Low_Bar"
    if 'high bar' in name or 'highbar' in name: return "High_Bar"
    if 'goblet' in name: return "Goblet_Squats"
    
    # Weight-based fallback (e.g., 100kg, 200lb)
    weight_match = re.search(r'(\d+)\s*(kg|lb)', name)
    if weight_match:
        return f"Heavy_{weight_match.group(1)}{weight_match.group(2)}"
    
    return "General_Variations"

def execute_diverse_grouping():
    if not os.path.exists(RESULTS_DIR): return

    # --- PHASE 1: FLATTEN (Fixing the "Trying" and "For" mess) ---
    print("🧹 Cleaning up messy subgroups...")
    for root, dirs, files in os.walk(RESULTS_DIR, topdown=False):
        for name in dirs:
            if name.startswith("SubGroup_") or name.startswith("Group_"):
                path = os.path.join(root, name)
                for item in os.listdir(path):
                    try:
                        shutil.move(os.path.join(path, item), os.path.join(RESULTS_DIR, item))
                    except: pass
                shutil.rmtree(path)

    # --- PHASE 2: TIER 1 DIVERSITY ---
    all_folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]
    
    # Mapping folders to their diverse primary categories
    tier_1_map = {
        "Group_Performance_PRs": ["pr", "1rm", "max", "record", "double", "triple", "finally"],
        "Group_Olympic_Weightlifting": ["clean", "jerk", "snatch", "hookgrip", "olympic"],
        "Group_Technical_FormCheck": ["form", "depth", "critique", "review", "assessment", "tips"],
        "Group_Safety_and_Fails": ["fail", "bail", "injury", "scary", "dropped", "safety"]
    }

    processed = set()

    for category, keywords in tier_1_map.items():
        cat_path = os.path.join(RESULTS_DIR, category)
        
        matches = [f for f in all_folders if f not in processed and any(k in f.lower() for k in keywords)]
        if matches:
            os.makedirs(cat_path, exist_ok=True)
            for m in matches:
                # --- PHASE 3: TIER 2 SUBGROUPING ---
                sub_label = get_biomechanical_subgroup(m)
                sub_path = os.path.join(cat_path, f"SubGroup_{sub_label}")
                os.makedirs(sub_path, exist_ok=True)
                
                shutil.move(os.path.join(RESULTS_DIR, m), os.path.join(sub_path, m))
                processed.add(m)

    # Final Catch-all for anything else
    remaining = [f for f in all_folders if f not in processed]
    if remaining:
        misc_path = os.path.join(RESULTS_DIR, "Group_General_Training")
        for r in remaining:
            sub_label = get_biomechanical_subgroup(r)
            sub_path = os.path.join(misc_path, f"SubGroup_{sub_label}")
            os.makedirs(sub_path, exist_ok=True)
            shutil.move(os.path.join(RESULTS_DIR, r), os.path.join(sub_path, r))

    print("\n✨ Diverse grouping complete. Verbs like 'Trying' have been purged.")

if __name__ == "__main__":
    execute_diverse_grouping()