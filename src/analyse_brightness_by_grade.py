
import pandas as pd
from scipy.stats import spearmanr

# Load the measured dataset
input_file = "data/colour_analysis/chip_colour_measurements.csv"
output_file = "data/colour_analysis/brightness_by_grade_summary.csv"

df = pd.read_csv(input_file)

required_columns = [
    "image_name",
    "grade",
    "estimated_chip_brightness",
    "acrylamide_ug_per_kg"
]

# Check required columns
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Clean numeric data
df["estimated_chip_brightness"] = pd.to_numeric(
    df["estimated_chip_brightness"], errors="coerce"
)
df["acrylamide_ug_per_kg"] = pd.to_numeric(
    df["acrylamide_ug_per_kg"], errors="coerce"
)
df = df.dropna(subset=required_columns)

print("=" * 65)
print("GRADE-WISE BRIGHTNESS ANALYSIS")
print("=" * 65)
print("Total valid samples:", len(df))

results = []

for grade, group in df.groupby("grade", sort=False):
    brightness = group["estimated_chip_brightness"]
    acrylamide = group["acrylamide_ug_per_kg"]

    # Correlation needs at least 3 samples and variation in both variables
    if (
        len(group) >= 3
        and brightness.nunique() > 1
        and acrylamide.nunique() > 1
    ):
        rho, p_value = spearmanr(brightness, acrylamide)
    else:
        rho, p_value = float("nan"), float("nan")

    result = {
        "grade": grade,
        "sample_count": len(group),
        "mean_brightness": brightness.mean(),
        "mean_acrylamide_ug_per_kg": acrylamide.mean(),
        "median_brightness": brightness.median(),
        "median_acrylamide_ug_per_kg": acrylamide.median(),
        "spearman_correlation": rho,
        "spearman_p_value": p_value
    }
    results.append(result)

    print(f"\nGrade: {grade}")
    print("  Samples:", len(group))
    print(f"  Mean brightness: {brightness.mean():.2f}")
    print(f"  Mean acrylamide: {acrylamide.mean():.2f} µg/kg")

    if pd.isna(rho):
        print("  Correlation: Not available")
    else:
        print(f"  Spearman correlation: {rho:.4f}")
        print(f"  P-value: {p_value:.6g}")

summary = pd.DataFrame(results)
summary.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("Summary saved to:", output_file)
print("=" * 65)
