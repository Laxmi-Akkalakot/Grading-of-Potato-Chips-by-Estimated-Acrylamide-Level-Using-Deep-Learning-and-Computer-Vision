
from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images"
LABELS_FILE = ROOT / "data" / "labels.csv"
OUTPUT_FILE = ROOT / "data" / "colour_analysis" / "contact_sheet.jpg"

df = pd.read_csv(LABELS_FILE)

# Show a range of measured acrylamide levels.
df = df.sort_values("acrylamide_ug_per_kg")
selected = pd.concat([
    df.head(12),
    df.iloc[len(df)//2 - 6:len(df)//2 + 6],
    df.tail(12)
]).drop_duplicates("image_name")

thumb_w, thumb_h = 150, 145
label_h = 48
columns = 6
rows = math.ceil(len(selected) / columns)

sheet = Image.new(
    "RGB",
    (columns * thumb_w, rows * (thumb_h + label_h)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, (_, row) in enumerate(selected.iterrows()):
    path = IMAGE_DIR / str(row["image_name"])
    if not path.exists():
        continue

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_w - 10, thumb_h - 10))

        x = (i % columns) * thumb_w
        y = (i // columns) * (thumb_h + label_h)

        sheet.paste(im, (x + (thumb_w - im.width)//2, y))
        draw.text(
            (x + 3, y + thumb_h),
            str(row["image_name"])[:23],
            fill="black"
        )
        draw.text(
            (x + 3, y + thumb_h + 18),
            f'{row["grade"]}: {row["acrylamide_ug_per_kg"]:.0f} ug/kg',
            fill="black"
        )
    except Exception as e:
        print("Skipped:", path.name, e)

sheet.save(OUTPUT_FILE)
print("Contact sheet saved to:", OUTPUT_FILE)
