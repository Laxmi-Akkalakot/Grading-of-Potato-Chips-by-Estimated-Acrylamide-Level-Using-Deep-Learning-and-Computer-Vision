
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "colour_analysis" / "colour_observations.csv"

df = pd.read_csv(FILE)

print("Total rows:", len(df))
print("Unique images:", df["image_name"].nunique())

duplicates = df[df.duplicated("image_name", keep=False)]

if duplicates.empty:
    print("\nNo duplicate image names found.")
else:
    print("\nDuplicate image records:")
    print(
        duplicates[
            ["image_name", "acrylamide_ug_per_kg", "grade", "visual_colour"]
        ].sort_values("image_name").to_string(index=False)
    )

print("\nGrade counts:")
print(df.drop_duplicates("image_name")["grade"].value_counts().to_string())
