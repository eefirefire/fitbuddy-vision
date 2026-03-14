import pandas as pd

try:
    df = pd.read_csv("squat_features_augmented.csv")
    print("\n✅ SUCCESS! Found the file.")
    print("---------------------------------------")
    print("COPY THESE COLUMN NAMES FOR THE NEXT STEP:")
    print(list(df.columns))
    print("---------------------------------------")
except FileNotFoundError:
    print("❌ Error: Put 'squat_features_augmented.csv' in the backend folder first!")