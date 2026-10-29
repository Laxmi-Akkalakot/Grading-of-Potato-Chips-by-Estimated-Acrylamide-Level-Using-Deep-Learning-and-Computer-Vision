
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images"
CSV_FILE = ROOT / "data" / "colour_analysis" / "colour_acrylamide_analysis.csv"
OUT_DIR = ROOT / "data" / "colour_analysis"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_FILE)

# Candidate 1: Low-grade samples with the highest estimated brown-pixel percentage.
low_candidates = (
    df[df["grade"] == "Low"]
    .sort_values("brown_pixel_percent", ascending=False)
    .head(12)
)

# Candidate 2: High/Very High samples with the lowest estimated brown-pixel percentage.
high_candidates = (
    df[df["grade"].isin(["High", "Very High"])]
    .sort_values("brown_pixel_percent", ascending=True)
    .head(12)
)

selected = pd.concat([low_candidates, high_candidates])
selected = selected.drop_duplicates("image_name").reset_index(drop=True)

cell_w, image_h, label_h = 300, 240, 58
columns = 3
rows = (len(selected) + columns - 1) // columns

sheet = Image.new(
    "RGB", (columns * cell_w, rows * (image_h + label_h)), "white"
)
draw = ImageDraw.Draw(sheet)

for i, row in selected.iterrows():
    path = IMAGE_DIR / str(row["image_name"])
    if not path.exists():
        continue

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((cell_w - 16, image_h - 12))

        x = (i % columns) * cell_w
        y = (i // columns) * (image_h + label_h)

        sheet.paste(im, (x + (cell_w - im.width) // 2, y))
        draw.text(
            (x + 6, y + image_h),
            str(row["image_name"])[:38],
            fill="black"
        )
        draw.text(
            (x + 6, y + image_h + 22),
            f'{row["grade"]} | Lab: {row["acrylamide_ug_per_kg"]:.1f} ug/kg',
            fill="black"
        )
    except Exception as error:
        print("Skipped:", path.name, error)

output = OUT_DIR / "candidate_images.jpg"
sheet.save(output)

print("Candidate sheet saved to:", output)
print("\nSelected candidates:")
print(selected[[
    "image_name", "grade", "acrylamide_ug_per_kg",
    "brown_pixel_percent"
]].to_string(index=False))
print("\nNote: brown-pixel percentage is only a rough image-level estimate.")
