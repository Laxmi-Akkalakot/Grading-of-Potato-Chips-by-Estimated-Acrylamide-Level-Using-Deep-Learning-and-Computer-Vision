from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "colour_analysis" / "colour_observations.csv"
OUTPUT = ROOT / "data" / "colour_analysis" / "colour_observation_summary.csv"

df = pd.read_csv(FILE)

# Remove completely empty rows and identify duplicate images.
df = df.dropna(how="all")

print("Total observation rows:", len(df))
print("Unique image names:", df["image_name"].nunique())

duplicates = df[df.duplicated("image_name", keep=False)]
if not duplicates.empty:
    print("\nDuplicate image records:")
    print(duplicates[["image_name", "grade"]].to_string(index=False))
else:
    print("\nNo duplicate image records.")

# Check missing visual observations.
missing = df[
    df["visual_colour"].isna()
    | (df["visual_colour"].astype(str).str.strip() == "")
]
print("\nImages needing visual observations:", len(missing))
if not missing.empty:
    print(missing["image_name"].to_string(index=False))

# Summarize the recorded colours by measured grade.
complete = df[
    df["visual_colour"].notna()
    & (df["visual_colour"].astype(str).str.strip() != "")
].copy()

summary = (
    complete.groupby(["grade", "visual_colour"])
    .size()
    .reset_index(name="image_count")
)

summary.to_csv(OUTPUT, index=False)

print("\nColour observations by measured grade:")
print(summary.to_string(index=False))
print("\nSummary saved to:", OUTPUT)
print(
    "\nThis is a descriptive summary, not proof that colour "
    "causes or accurately predicts acrylamide levels."
)