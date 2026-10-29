
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "colour_analysis" / "colour_observations.csv"
OUTPUT_DIR = ROOT / "data" / "colour_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Remove rows without a visual description.
df["visual_colour"] = df["visual_colour"].fillna("").str.strip()
df = df[df["visual_colour"] != ""].copy()

print("TOTAL VISUALLY DESCRIBED IMAGES:", len(df))
print("\nSAMPLES BY MEASURED GRADE:")
print(df["grade"].value_counts().to_string())

# Create a descriptive summary of the recorded colour labels.
summary = (
    df.groupby(["grade", "visual_colour"])
      .size()
      .reset_index(name="image_count")
      .sort_values(["grade", "image_count"], ascending=[True, False])
)

summary_file = OUTPUT_DIR / "final_colour_grade_summary.csv"
summary.to_csv(summary_file, index=False)

print("\nCOLOUR DESCRIPTIONS BY GRADE:")
print(summary.to_string(index=False))

# List samples that have similar broad colour descriptions.
print("\nSAMPLES WITH GOLDEN/YELLOW DESCRIPTIONS:")
golden = df[
    df["visual_colour"].str.contains(
        r"golden|yellow|pale", case=False, regex=True
    )
]
print(
    golden[
        ["image_name", "acrylamide_ug_per_kg", "grade", "visual_colour"]
    ].to_string(index=False)
)

print("\nSAMPLES WITH BROWN DESCRIPTIONS:")
brown = df[
    df["visual_colour"].str.contains("brown", case=False, regex=False)
]
print(
    brown[
        ["image_name", "acrylamide_ug_per_kg", "grade", "visual_colour"]
    ].to_string(index=False)
)

print("\nREPORT SAVED TO:", summary_file)
print(
    "\nNOTE: This report summarizes visual descriptions. "
    "It does not establish that colour causes or reliably predicts "
    "acrylamide concentration."
)
