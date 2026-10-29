
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "colour_analysis" / "colour_observations.csv"
OUTPUT_DIR = ROOT / "data" / "colour_analysis"

df = pd.read_csv(INPUT_FILE)
df["visual_colour"] = df["visual_colour"].fillna("").str.strip()
df["grade"] = df["grade"].fillna("").str.strip()

# Display the number of observations in each measured grade.
print("\n1. Observations by measured grade")
print(df["grade"].value_counts().to_string())

# Find records whose descriptions mention pale/light or dark/brown.
text = df["visual_colour"].str.lower()

df["mentions_pale_or_light"] = text.str.contains(
    r"\bpale\b|\blight\b", regex=True
)
df["mentions_dark_or_brown"] = text.str.contains(
    r"\bdark\b|\bbrown\b", regex=True
)

print("\n2. Pale/light descriptions by measured grade")
print(
    pd.crosstab(
        df["grade"],
        df["mentions_pale_or_light"]
    ).to_string()
)

print("\n3. Dark/brown descriptions by measured grade")
print(
    pd.crosstab(
        df["grade"],
        df["mentions_dark_or_brown"]
    ).to_string()
)

# Save records that deserve closer visual inspection.
candidates = df[
    (
        df["mentions_pale_or_light"]
        & df["grade"].isin(["High", "Very High"])
    )
    |
    (
        df["mentions_dark_or_brown"]
        & (df["grade"] == "Low")
    )
].copy()

output = OUTPUT_DIR / "visual_colour_exception_candidates.csv"
candidates.to_csv(output, index=False)

print("\n4. Candidate records requiring manual inspection")
if candidates.empty:
    print("No candidates found using these description keywords.")
else:
    print(candidates[
        ["image_name", "acrylamide_ug_per_kg",
         "grade", "visual_colour", "notes"]
    ].to_string(index=False))

print("\nSaved candidate file:", output)
print(
    "\nImportant: keyword matches are only candidates. "
    "They do not prove that the chip is uniformly pale or dark."
)
