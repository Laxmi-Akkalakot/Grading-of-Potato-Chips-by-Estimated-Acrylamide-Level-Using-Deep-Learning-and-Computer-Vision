

from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images"
OUTPUT_FILE = (
    ROOT / "data" / "colour_analysis"
    / "misclassified_test_images.jpg"
)

# Misclassified test images from check_predictions.py
errors = [
    {
        "image": "Ch3_067_Medium_627.jpg",
        "actual": "Medium",
        "predicted": "Very High",
        "confidence": 42.41,
    },
    {
        "image": "Ch3_304_Very-high_6526.jpg",
        "actual": "Very High",
        "predicted": "High",
        "confidence": 46.43,
    },
    {
        "image": "Ch3_131_Medium_851.jpg",
        "actual": "Medium",
        "predicted": "Low",
        "confidence": 72.22,
    },
    {
        "image": "Ch3_296_High_1310.jpg",
        "actual": "High",
        "predicted": "Very High",
        "confidence": 41.05,
    },
    {
        "image": "Ch3_035_Medium_632.jpg",
        "actual": "Medium",
        "predicted": "Very High",
        "confidence": 41.05,
    },
    {
        "image": "Ch3_199_Medium_809.jpg",
        "actual": "Medium",
        "predicted": "High",
        "confidence": 46.23,
    },
    {
        "image": "Ch3_297_Medium_529.jpg",
        "actual": "Medium",
        "predicted": "Very High",
        "confidence": 42.41,
    },
]

thumb_w, thumb_h = 260, 220
label_h = 72
columns = 3
rows = math.ceil(len(errors) / columns)

sheet = Image.new(
    "RGB",
    (columns * thumb_w, rows * (thumb_h + label_h)),
    "white",
)
draw = ImageDraw.Draw(sheet)

for i, item in enumerate(errors):
    path = IMAGE_DIR / item["image"]

    x = (i % columns) * thumb_w
    y = (i // columns) * (thumb_h + label_h)

    if not path.exists():
        draw.text(
            (x + 8, y + 8),
            f'Image not found: {item["image"]}',
            fill="red",
        )
        continue

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_w - 16, thumb_h - 16))

        paste_x = x + (thumb_w - im.width) // 2
        paste_y = y + (thumb_h - im.height) // 2
        sheet.paste(im, (paste_x, paste_y))

        draw.text(
            (x + 8, y + thumb_h),
            f'Image: {item["image"][:30]}',
            fill="black",
        )
        draw.text(
            (x + 8, y + thumb_h + 18),
            f'Actual: {item["actual"]}',
            fill="black",
        )
        draw.text(
            (x + 8, y + thumb_h + 36),
            f'Predicted: {item["predicted"]}',
            fill="black",
        )
        draw.text(
            (x + 8, y + thumb_h + 54),
            f'Confidence: {item["confidence"]:.2f}%',
            fill="black",
        )

    except Exception as e:
        print("Skipped:", path.name, e)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT_FILE)

print("Contact sheet saved to:", OUTPUT_FILE)
