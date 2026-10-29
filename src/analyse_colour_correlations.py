
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# Load the measured dataset
file_path = "data/colour_analysis/chip_colour_measurements.csv"
df = pd.read_csv(file_path)

# Colour features to analyse
features = [
    "chip_mean_red",
    "chip_mean_green",
    "chip_mean_blue",
    "estimated_chip_brightness"
]

target = "acrylamide_ug_per_kg"

print("=" * 65)
print("COLOUR FEATURES VS LABORATORY ACRYLAMIDE")
print("=" * 65)

print("Total samples:", len(df))
print("Missing values in acrylamide:", df[target].isna().sum())

# Calculate correlations
results = []

for feature in features:
    valid = df[[feature, target]].dropna()

    if len(valid) < 3:
        print(f"\nSkipping {feature}: insufficient valid samples.")
        continue

    spearman_r, spearman_p = spearmanr(
        valid[feature], valid[target]
    )
    pearson_r, pearson_p = pearsonr(
        valid[feature], valid[target]
    )

    results.append({
        "feature": feature,
        "sample_count": len(valid),
        "spearman_correlation": spearman_r,
        "spearman_p_value": spearman_p,
        "pearson_correlation": pearson_r,
        "pearson_p_value": pearson_p
    })

    print(f"\nFeature: {feature}")
    print(f"  Samples: {len(valid)}")
    print(f"  Spearman correlation: {spearman_r:.4f}")
    print(f"  Spearman p-value: {spearman_p:.6f}")
    print(f"  Pearson correlation:  {pearson_r:.4f}")
    print(f"  Pearson p-value:      {pearson_p:.6f}")

# Save results
results_df = pd.DataFrame(results)
output_path = "data/colour_analysis/colour_correlation_results.csv"
results_df.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("Analysis saved to:", output_path)
print("=" * 65)
