
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "labels.csv")

missing_files = []
unreadable_files = []

for _, row in df.iterrows():
    image_path = ROOT / row["image_path"]

    if not image_path.exists():
        missing_files.append(str(image_path))
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception:
        unreadable_files.append(str(image_path))

print("Total CSV rows:", len(df))
print("Missing image files:", len(missing_files))
print("Unreadable image files:", len(unreadable_files))

if missing_files:
    print("\nFirst missing files:")
    for path in missing_files[:10]:
        print(path)

if unreadable_files:
    print("\nFirst unreadable files:")
    for path in unreadable_files[:10]:
        print(path)
