
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images"
OUTPUT_DIR = ROOT / "data" / "colour_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use a single-chip image for the first test.
IMAGE_NAME = "Ch3_046_Low_133.jpg"
IMAGE_PATH = IMAGE_DIR / IMAGE_NAME

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise ValueError("OpenCV could not read the image.")

height, width = image.shape[:2]

# Rectangle around the central chip.
# GrabCut will try to separate the chip from the background.
x = int(width * 0.15)
y = int(height * 0.15)
w = int(width * 0.70)
h = int(height * 0.70)

mask = np.zeros((height, width), np.uint8)
background_model = np.zeros((1, 65), np.float64)
foreground_model = np.zeros((1, 65), np.float64)

cv2.grabCut(
    image,
    mask,
    (x, y, w, h),
    background_model,
    foreground_model,
    5,
    cv2.GC_INIT_WITH_RECT
)

# Foreground pixels are labelled 1 or 3 by GrabCut.
chip_mask = np.where(
    (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
    255,
    0
).astype(np.uint8)

# Remove small unwanted regions and keep the largest connected region.
count, components, stats, centroids = cv2.connectedComponentsWithStats(
    chip_mask, connectivity=8
)

if count > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    chip_mask = np.where(components == largest_label, 255, 0).astype(np.uint8)

# Shrink the mask slightly to reduce background and shadow near the edge.
kernel = np.ones((5, 5), np.uint8)
chip_mask = cv2.erode(chip_mask, kernel, iterations=2)

# Highlight the estimated chip pixels in red.
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
overlay = rgb.copy()
red = np.zeros_like(overlay)
red[:, :, 0] = 255

selected = chip_mask == 255
overlay[selected] = (
    0.55 * overlay[selected] + 0.45 * red[selected]
).astype(np.uint8)

# Put original and mask preview side by side.
original_pil = Image.fromarray(rgb)
overlay_pil = Image.fromarray(overlay)

original_pil.thumbnail((400, 400))
overlay_pil.thumbnail((400, 400))

canvas = Image.new("RGB", (800, 440), "white")
canvas.paste(original_pil, (0, 0))
canvas.paste(overlay_pil, (400, 0))

draw = ImageDraw.Draw(canvas)
draw.text((10, 410), "Original image", fill="black")
draw.text((410, 410), "GrabCut estimate in red", fill="black")


# Calculate colour statistics using the estimated chip mask.
chip_pixels = rgb[chip_mask == 255]

if len(chip_pixels) > 0:
    mean_rgb = chip_pixels.mean(axis=0)

    brightness = (
        0.2126 * chip_pixels[:, 0]
        + 0.7152 * chip_pixels[:, 1]
        + 0.0722 * chip_pixels[:, 2]
    ).mean()

    print("Estimated chip mean RGB:", mean_rgb.round(2).tolist())
    print("Estimated chip brightness:", round(float(brightness), 2))
else:
    print("No chip pixels detected. Do not use this mask for measurements.")


output_path = OUTPUT_DIR / "chip_mask_preview.jpg"
canvas.save(output_path)

print("Image:", IMAGE_NAME)
print("Saved preview:", output_path)
print("Inspect the red area before using the mask for measurements.")
