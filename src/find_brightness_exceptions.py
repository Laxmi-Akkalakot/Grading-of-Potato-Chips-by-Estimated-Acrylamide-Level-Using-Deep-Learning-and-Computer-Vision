
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images"
CSV_FILE = ROOT / "data" / "colour_analysis" / "chip_colour_measurements.csv"
OUT_DIR = ROOT / "data" / "colour_analysis"

df = pd.read_csv(CSV_FILE)

# Possible dark-looking, Low-acrylamide examples.
dark_low = (
    df[df["grade"] == "Low"]
    .sort_values("estimated_chip_brightness", ascending=True)
    .head(8)
)

# Possible light-looking, High/Very High-acrylamide examples.
light_high = (
    df[df["grade"].isin(["High", "Very High"])]
    .sort_values("estimated_chip_brightness", ascending=False)
    .head(8)
)

selected = pd.concat([dark_low, light_high])
selected = selected.drop_duplicates("image_name").reset_index(drop=True)

cell_w, image_h, label_h = 300, 240, 55
columns = 4
rows = (len(selected) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (columns * cell_w, rows * (image_h + label_h)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, row in selected.iterrows():
    path = IMAGE_DIR / str(row["image_name"])
    if not path.exists():
        continue

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((cell_w - 12, image_h - 10))

    x = (i % columns) * cell_w
    y = (i // columns) * (image_h + label_h)

    sheet.paste(im, (x + (cell_w - im.width) // 2, y))
    draw.text(
        (x + 5, y + image_h),
        str(row["image_name"])[:38],
        fill="black"
    )
    draw.text(
        (x + 5, y + image_h + 22),
        f'{row["grade"]} | {row["acrylamide_ug_per_kg"]:.1f} ug/kg',
        fill="black"
    )

output = OUT_DIR / "brightness_exception_candidates.jpg"
sheet.save(output)

print("Candidate images:", len(selected))
print("Saved to:", output)
print("\nCandidate records:")
print(selected[[
    "image_name",
    "grade",
    "acrylamide_ug_per_kg",
    "estimated_chip_brightness"
]].to_string(index=False))

print("\nThese are candidates, not confirmed colour exceptions.")
