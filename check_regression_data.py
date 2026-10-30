
import pandas as pd
import os

# ==========================================
# PART 1: Check Original Dataset
# ==========================================

print("=" * 50)
print("ORIGINAL DATASET CHECK")
print("=" * 50)

df = pd.read_csv("data/labels.csv")

# Check whether image files exist
df["file_exists"] = df["image_path"].apply(os.path.exists)

print("Total records:", len(df))
print("Images found:", df["file_exists"].sum())
print("Images missing:", (~df["file_exists"]).sum())

# Check duplicate records
print("\nDuplicate image names:", df["image_name"].duplicated().sum())

# Check grade distribution
print("\nGrade distribution:")
print(df["grade"].value_counts())

# Check numerical values by grade
print("\nAcrylamide statistics by grade:")
print(
    df.groupby("grade")["acrylamide_ug_per_kg"]
    .agg(["count", "min", "max", "mean", "median"])
)


# ==========================================
# PART 2: Check Regression Dataset
# ==========================================

print("\n" + "=" * 50)
print("REGRESSION DATASET CHECK")
print("=" * 50)

regression_df = pd.read_csv("data/regression_labels.csv")

print("\nDataset shape:", regression_df.shape)

# Check missing values
print("\nMissing values:")
print(regression_df.isnull().sum())

# Check duplicate images
print("\nDuplicate images:")
print(regression_df["image_name"].duplicated().sum())

# Check whether image files exist
print("\nImages found:")
print(regression_df["image_path"].apply(os.path.exists).sum())

print("\nImages missing:")
print((~regression_df["image_path"].apply(os.path.exists)).sum())

# Display first five records
print("\nFirst 5 records:")
print(regression_df.head())

# Numerical statistics
print("\nAcrylamide statistics:")
print(regression_df["acrylamide_ug_per_kg"].describe())

# Grade-wise statistics
print("\nGrade-wise statistics:")
print(
    regression_df.groupby("grade")["acrylamide_ug_per_kg"]
    .agg(["count", "min", "max", "mean", "median"])
)

print("\nDataset checking completed!")
