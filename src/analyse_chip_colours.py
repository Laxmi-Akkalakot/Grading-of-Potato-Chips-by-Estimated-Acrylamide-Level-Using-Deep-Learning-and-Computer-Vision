
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images"
LABELS_FILE = ROOT / "data" / "labels.csv"
OUTPUT_DIR = ROOT / "data" / "colour_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LABELS_FILE)
results = []

print("Starting colour analysis...")
print("Total labelled images:", len(df))

for index, row in df.iterrows():
    name = str(row["image_name"])
    path = IMAGE_DIR / name

    if not path.exists():
        print(f"Image not found: {name}")
        continue

    image = cv2.imread(str(path))

    if image is None:
        print(f"Could not read: {name}")
        continue

    # Resize images to speed up segmentation.
    image = cv2.resize(image, (300, 300))

    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    # Central rectangle used as an initial foreground estimate.
    rect = (
        int(width * 0.15),
        int(height * 0.15),
        int(width * 0.70),
        int(height * 0.70),
    )

    try:
        cv2.grabCut(
            image,
            mask,
            rect,
            bg_model,
            fg_model,
            1,
            cv2.GC_INIT_WITH_RECT,
        )

        foreground = (
            (mask == cv2.GC_FGD)
            | (mask == cv2.GC_PR_FGD)
        )

        # Keep the largest connected foreground region.
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8),
            connectivity=8,
        )

        if count <= 1:
            continue

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        foreground = components == largest

        # Slightly shrink the selected area to reduce edge contamination.
        foreground = cv2.erode(
            foreground.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        ).astype(bool)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = rgb[foreground]

        if len(pixels) < 100:
            continue

        brightness = (
            0.2126 * pixels[:, 0]
            + 0.7152 * pixels[:, 1]
            + 0.0722 * pixels[:, 2]
        ).mean()

        results.append({
            "image_name": name,
            "acrylamide_ug_per_kg": float(
                row["acrylamide_ug_per_kg"]
            ),
            "grade": row["grade"],
            "chip_mean_red": float(pixels[:, 0].mean()),
            "chip_mean_green": float(pixels[:, 1].mean()),
            "chip_mean_blue": float(pixels[:, 2].mean()),
            "estimated_chip_brightness": float(brightness),
        })

    except cv2.error as error:
        print(f"Could not analyse {name}: {error}")

    # Show progress every 25 images.
    if (index + 1) % 25 == 0:
        print(f"Processed {index + 1}/{len(df)} images...")

output = OUTPUT_DIR / "chip_colour_measurements.csv"
pd.DataFrame(results).to_csv(output, index=False)

print("\nAnalysis finished.")
print("Images in labels.csv:", len(df))
print("Images analysed:", len(results))
print("Results saved to:", output)
print(
    "Note: segmentation is approximate. Check masks before "
    "using the measurements for scientific conclusions."
)
