import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("🧠 Training the Fit-Buddy Angle Brain...")

try:
    data = pd.read_csv('squat_features_augmented.csv')
except FileNotFoundError:
    print("❌ Error: CSV not found!")
    exit()

# --- THE FIX: Select ONLY the columns we can calculate live ---
# We ignore 'video_file', 'frame', 'symmetry_score', etc.
selected_features = [
    'left_knee_angle', 
    'right_knee_angle', 
    'left_hip_angle', 
    'right_hip_angle', 
    'torso_lean'  # Usually called 'spine_angle' or similar in datasets
]

# Check if 'torso_lean' exists, otherwise try 'spine_angle'
if 'torso_lean' not in data.columns:
    if 'spine_angle' in data.columns:
        selected_features[-1] = 'spine_angle'
    else:
        print("⚠️ Warning: Could not find lean/spine angle. Removing it.")
        selected_features.pop()

print(f"ℹ️ Training on these {len(selected_features)} features: {selected_features}")

# Filter the data
X = data[selected_features]
y = data.iloc[:, -1] # The label is still the last column

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save properly
artifact = {
    "model": model,
    "features": selected_features
}

with open('fitbuddy_angle_brain.pkl', 'wb') as f:
    pickle.dump(artifact, f)
    
print("💾 Saved 'fitbuddy_angle_brain.pkl'")