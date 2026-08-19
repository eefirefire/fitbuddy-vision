import io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. READ AND CLEAN THE DATA LAYER
filename = "FitBuddy Valdiation Sheet - Rehab.csv"

if not os.path.exists(filename):
    raise FileNotFoundError(
        f"Could not find '{filename}'. Please make sure this script is in the same folder as the CSV file."
    )

# Handle potential metadata headers or byte markers in the exported CSV
with open(filename, "rb") as f:
    content = f.read()

start_idx = content.find(b"REHAB,Reps")
if start_idx == -1:
    # Fallback if standard parsing works
    csv_part = content
else:
    csv_part = content[start_idx:]

# Remove trailing metadata or links if present
end_idx = csv_part.find(b'"Bhttps')
if end_idx != -1:
    csv_part = csv_part[:end_idx]

# Parse clean DataFrame
df = pd.read_csv(io.BytesIO(csv_part), encoding="utf-8")

# Forward fill the 'REHAB' categories and drop empty spacer rows
df["REHAB"] = df["REHAB"].ffill()
df = df.dropna(subset=["Reps"])

# Clean string formatting anomalies
df["REHAB"] = df["REHAB"].str.replace("Off\\_Angle", "Off_Angle", regex=False)

# Strip any accidental leading or trailing spaces from column names
df.columns = df.columns.str.strip()

# Ensure matching types across target parameters safely
df["Online protactor angle"] = pd.to_numeric(df["Online protactor angle"])
df["Fitbuddy Angle"] = pd.to_numeric(df["Fitbuddy Angle"])
df["Error"] = pd.to_numeric(df["Error"])  # <-- Cleaned up the spacing here
df["Abs_Error"] = df["Error"].abs()


# --- GRAPH 1: ACCURACY CORRELATION PLOT ---
print("Generating Accuracy Correlation Plot...")
aligned_df = df[df["REHAB"].isin(["Perfect", "Faulty"])]

plt.figure(figsize=(7, 6), dpi=300)
sns.scatterplot(
    data=aligned_df,
    x="Online protactor angle",
    y="Fitbuddy Angle",
    hue="REHAB",
    style="REHAB",
    s=120,
    palette=["#2ca02c", "#ff7f0e"],
)

# Calculate linear regression trendline and exact R² coefficient
m, b = np.polyfit(
    aligned_df["Online protactor angle"], aligned_df["Fitbuddy Angle"], 1
)
r_squared = (
    np.corrcoef(
        aligned_df["Online protactor angle"], aligned_df["Fitbuddy Angle"]
    )[0, 1]
    ** 2
)

# Render trendline boundaries
x_vals = np.linspace(130, 175, 100)
plt.plot(
    x_vals,
    m * x_vals + b,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=f"Trendline ($R^2 = {r_squared:.3f}$)",
)

plt.title(
    "System Accuracy Under Optimal Alignment", fontsize=13, fontweight="bold"
)
plt.xlabel("Ground Truth Angle (Online Protractor, Degrees)", fontsize=11)
plt.ylabel("FitBuddy Detected Angle (Degrees)", fontsize=11)
plt.xlim(128, 177)
plt.ylim(135, 173)
plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("rehab_accuracy_correlation.png", dpi=300)
plt.close()


# --- GRAPH 2: ENVIRONMENTAL CONSTRAINT IMPACT BAR CHART ---
print("Generating Mean Absolute Error Chart...")
mae_summary = df.groupby("REHAB")["Abs_Error"].mean().reset_index()

# Sort groups explicitly to match standard presentation hierarchy
mae_summary["REHAB"] = pd.Categorical(
    mae_summary["REHAB"],
    categories=["Perfect", "Faulty", "Off_Angle"],
    ordered=True,
)
mae_summary = mae_summary.sort_values("REHAB")

plt.figure(figsize=(6.5, 5), dpi=300)
bars = plt.bar(
    mae_summary["REHAB"],
    mae_summary["Abs_Error"],
    color=["#2ca02c", "#ff7f0e", "#d62728"],
    edgecolor="black",
    alpha=0.8,
    width=0.45,
)

# Draw precise value metrics over every category block
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.2,
        f"{yval:.2f}°",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

plt.title(
    "Mean Absolute Error (MAE) Across Conditions",
    fontsize=13,
    fontweight="bold",
)
plt.ylabel("Mean Absolute Error (Degrees)", fontsize=11)
plt.ylim(0, 8.0)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("mae_by_condition.png", dpi=300)
plt.close()

print("\nSuccess! Two images have been exported cleanly to your directory:")
print(" 1. 'rehab_accuracy_correlation.png'")
print(" 2. 'mae_by_condition.png'")
