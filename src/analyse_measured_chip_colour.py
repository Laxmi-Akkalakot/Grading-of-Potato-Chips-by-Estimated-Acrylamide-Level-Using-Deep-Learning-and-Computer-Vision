
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "colour_analysis" / "chip_colour_measurements.csv"
OUTPUT_DIR = ROOT / "data" / "colour_analysis"

df = pd.read_csv(DATA_FILE)

if df.empty:
    raise ValueError("The measurements file is empty.")

# Calculate the correlation between estimated brightness and
# laboratory-measured acrylamide concentration.
correlation = df["estimated_chip_brightness"].corr(
    df["acrylamide_ug_per_kg"],
    method="spearman"
)

print("Images included:", len(df))
print("Spearman correlation:", round(correlation, 3))

plt.figure(figsize=(9, 6))

for grade, group in df.groupby("grade"):
    plt.scatter(
        group["estimated_chip_brightness"],
        group["acrylamide_ug_per_kg"],
        label=grade,
        alpha=0.7
    )

plt.xlabel("Estimated chip brightness (0-255)")
plt.ylabel("Measured acrylamide (µg/kg)")
plt.title("Estimated Chip Brightness vs Measured Acrylamide")
plt.legend(title="Measured grade")
plt.tight_layout()

plot_path = OUTPUT_DIR / "chip_brightness_vs_acrylamide.png"
plt.savefig(plot_path, dpi=160)
plt.close()

# Save a summary table by measured grade.
summary = df.groupby("grade")["estimated_chip_brightness"].agg(
    ["count", "mean", "median", "std"]
)

summary_path = OUTPUT_DIR / "chip_brightness_summary.csv"
summary.to_csv(summary_path)

print("Graph saved to:", plot_path)
print("Summary saved to:", summary_path)
print("\nBrightness summary by grade:")
print(summary.round(2))
print(
    "\nCaution: these results are preliminary because automatic "
    "segmentation may include shadows or background."
)
