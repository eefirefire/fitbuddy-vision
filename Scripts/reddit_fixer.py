import requests
import os
import time

# Configuration
COACH_WORDS = ["depth", "knees", "hips", "back", "heels", "stance", "lower", "drive", "form", "bar", "weight", "lean", "feet", "parallel", "ankle", "mobility", "chest", "upright", "butt", "wink"]
BOT_BLACKLIST = ["i am a bot", "action was performed automatically", "unsolicited advice"]
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def find_missing_url(title):
    """Attempts to find the Reddit URL using the post title."""
    search_url = f"https://www.reddit.com/search.json?q={title}&limit=1"
    try:
        response = requests.get(search_url, headers=HEADERS)
        data = response.json()
        post_data = data['data']['children'][0]['data']
        return f"https://www.reddit.com{post_data['permalink']}"
    except Exception:
        return None

def get_better_feedback(url):
    """Fetches human comments only, skipping bots."""
    json_url = url.rstrip('/') + ".json"
    feedback = []
    try:
        response = requests.get(json_url, headers=HEADERS)
        data = response.json()
        # Comments are in the second object of the Reddit JSON response
        comments = data[1]['data']['children']
        for c in comments:
            if c['kind'] == 't1':
                body = c['data'].get('body', '')
                low_body = body.lower()
                
                # Logic: No bots + No short comments + (Keywords OR Depth)
                is_bot = any(b in low_body for b in BOT_BLACKLIST)
                is_long = len(body.split()) > 20
                has_keyword = any(w in low_body for w in COACH_WORDS)
                
                if not is_bot and len(body.split()) > 5:
                    if has_keyword or is_long:
                        clean_text = body.strip().replace('\n', ' ')
                        feedback.append(f"- {clean_text}")
    except Exception:
        pass
    return feedback

def fix_dataset(directory="data/squats"):
    if not os.path.exists(directory):
        print(f"Directory {directory} not found!")
        return

    files = [f for f in os.listdir(directory) if f.endswith('_feedback.txt')]
    print(f"Auditing {len(files)} files...")

    for filename in files:
        filepath = os.path.join(directory, filename)
        url, title, action = None, None, "00:00 - End"
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith("TITLE:"): title = line.replace("TITLE:", "").strip()
                if line.startswith("URL:"): url = line.replace("URL:", "").strip()
                if line.startswith("ESTIMATED ACTION:"): action = line.replace("ESTIMATED ACTION:", "").strip()

            # Rescue mission for missing URLs
            if not url and title:
                print(f"Searching for URL: {title[:30]}...")
                url = find_missing_url(title)
                time.sleep(2)

            if url:
                print(f"Updating: {title[:30]}...")
                new_comments = get_better_feedback(url)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"TITLE: {title}\nURL: {url}\nESTIMATED ACTION: {action}\n\nFEEDBACK:\n")
                    if new_comments:
                        f.write("\n".join(new_comments))
                    else:
                        f.write("[No human feedback found]")
                time.sleep(5)
            else:
                print(f"FAILED: No URL found for {filename}")
        except Exception as e:
            print(f"System Error on {filename}: {e}")

if __name__ == "__main__":
    fix_dataset()