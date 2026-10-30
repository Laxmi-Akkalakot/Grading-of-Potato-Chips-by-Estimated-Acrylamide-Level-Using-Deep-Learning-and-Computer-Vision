
import os
import pandas as pd

# Load existing verified dataset
df = pd.read_csv("data/labels.csv")

# Required columns
required_columns = [
    "image_name",
    "image_path",
    "acrylamide_ug_per_kg",
    "grade"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

# Check image availability
df["file_exists"] = df["image_path"].apply(os.path.exists)

if not df["file_exists"].all():
    raise FileNotFoundError("Some image files are missing.")

# Check numerical target
df["acrylamide_ug_per_kg"] = pd.to_numeric(
    df["acrylamide_ug_per_kg"],
    errors="coerce"
)

if df["acrylamide_ug_per_kg"].isna().any():
    raise ValueError("Invalid acrylamide values found.")

# Check duplicate image names
if df["image_name"].duplicated().any():
    raise ValueError("Duplicate image names found.")

# Keep only required columns
regression_df = df[
    required_columns
].copy()

# Save prepared dataset
output_path = "data/regression_labels.csv"

regression_df.to_csv(output_path, index=False)

print("Regression dataset prepared successfully!")
print("Total images:", len(regression_df))
print("Output:", output_path)

print("\nGrade distribution:")
print(regression_df["grade"].value_counts())

print("\nAcrylamide value summary:")
print(regression_df["acrylamide_ug_per_kg"].describe())
