
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. PROJECT PATHS
# ==========================================
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images"
LABELS_FILE = ROOT / "data" / "labels.csv"
OUTPUT_DIR = ROOT / "data" / "colour_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. CHECK DATASET
# ==========================================
if not LABELS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find labels file: {LABELS_FILE}"
    )

df = pd.read_csv(LABELS_FILE)

required = [
    "image_name",
    "acrylamide_ug_per_kg",
    "grade",
]

missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing columns in labels.csv: {missing}")

print("Labelled records:", len(df))
print("Images folder:", IMAGE_DIR)

# ==========================================
# 3. EXTRACT APPROXIMATE COLOUR FEATURES
# ==========================================
results = []

for _, row in df.iterrows():
    image_name = str(row["image_name"])
    image_path = IMAGE_DIR / image_name

    if not image_path.exists():
        continue

    try:
        with Image.open(image_path) as im:
            rgb = np.asarray(
                im.convert("RGB").resize((200, 200)),
                dtype=np.float32,
            )

        # Approximate perceived brightness.
        brightness = (
            0.2126 * rgb[:, :, 0]
            + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2]
        )

        # Approximate brown-like pixels.
        brown_mask = (
            (rgb[:, :, 0] > rgb[:, :, 1])
            & (rgb[:, :, 1] > rgb[:, :, 2])
            & (rgb[:, :, 0] < 190)
            & (rgb[:, :, 1] < 150)
            & (rgb[:, :, 2] < 120)
        )

        results.append({
            "image_name": image_name,
            "acrylamide_ug_per_kg": float(
                row["acrylamide_ug_per_kg"]
            ),
            "grade": str(row["grade"]),
            "mean_brightness": float(np.mean(brightness)),
            "brown_pixel_percent": float(
                np.mean(brown_mask) * 100
            ),
        })

    except Exception as error:
        print(f"Skipped {image_name}: {error}")

analysis = pd.DataFrame(results)

if analysis.empty:
    raise ValueError("No labelled images could be analysed.")

# ==========================================
# 4. SAVE COLOUR ANALYSIS
# ==========================================
csv_path = OUTPUT_DIR / "colour_acrylamide_analysis.csv"
analysis.to_csv(csv_path, index=False)

print("\nAnalysed images:", len(analysis))
print("\nGrade counts:")
print(analysis["grade"].value_counts())

# ==========================================
# 5. EXPLORE LIGHTER AND DARKER SAMPLES
# ==========================================
# Use relative groups from this dataset, not arbitrary
# claims that a colour determines acrylamide content.
light_cutoff = analysis["mean_brightness"].quantile(0.25)
dark_cutoff = analysis["mean_brightness"].quantile(0.75)

analysis["brightness_group"] = np.select(
    [
        analysis["mean_brightness"] <= light_cutoff,
        analysis["mean_brightness"] >= dark_cutoff,
    ],
    [
        "Lighter quarter",
        "Darker quarter",
    ],
    default="Middle half",
)

summary = (
    analysis.groupby(["brightness_group", "grade"])
    .size()
    .unstack(fill_value=0)
)

summary_path = OUTPUT_DIR / "brightness_grade_counts.csv"
summary.to_csv(summary_path)

print("\nBrightness group cutoffs:")
print(f"Lighter quarter: brightness <= {light_cutoff:.2f}")
print(f"Darker quarter: brightness >= {dark_cutoff:.2f}")

print("\nGrade counts by image brightness group:")
print(summary)

# ==========================================
# 6. LIST REAL SAMPLES FOR MANUAL INSPECTION
# ==========================================
# The brightness groups are approximate and may be affected
# by the background. Inspect these images before describing
# them as pale or dark potato chips.
light_samples = analysis[
    analysis["brightness_group"] == "Lighter quarter"
].sort_values("acrylamide_ug_per_kg", ascending=False)

dark_samples = analysis[
    analysis["brightness_group"] == "Darker quarter"
].sort_values("acrylamide_ug_per_kg", ascending=True)

light_samples.to_csv(
    OUTPUT_DIR / "lighter_images_sorted_by_acrylamide.csv",
    index=False,
)

dark_samples.to_csv(
    OUTPUT_DIR / "darker_images_sorted_by_acrylamide.csv",
    index=False,
)

print("\nLighter images with the highest measured values:")
print(
    light_samples[
        ["image_name", "acrylamide_ug_per_kg", "grade",
         "mean_brightness"]
    ].head(10).to_string(index=False)
)

print("\nDarker images with the lowest measured values:")
print(
    dark_samples[
        ["image_name", "acrylamide_ug_per_kg", "grade",
         "mean_brightness"]
    ].head(10).to_string(index=False)
)

# ==========================================
# 7. PLOT BRIGHTNESS AGAINST MEASURED VALUE
# ==========================================
plt.figure(figsize=(9, 6))

for grade, group in analysis.groupby("grade"):
    plt.scatter(
        group["mean_brightness"],
        group["acrylamide_ug_per_kg"],
        label=grade,
        alpha=0.75,
    )

plt.xlabel("Approximate whole-image brightness (0-255)")
plt.ylabel("Measured acrylamide (µg/kg)")
plt.title("Image Brightness vs Measured Acrylamide")
plt.legend(title="Measured grade")
plt.tight_layout()

plot_path = OUTPUT_DIR / "brightness_vs_acrylamide.png"
plt.savefig(plot_path, dpi=160)
plt.close()

print("\nAnalysis complete.")
print("Saved:", csv_path)
print("Saved:", summary_path)
print("Saved:", plot_path)
print("Output folder:", OUTPUT_DIR)
print(
    "\nImportant: brightness includes the image background. "
    "Inspect the images before drawing scientific conclusions."
)
