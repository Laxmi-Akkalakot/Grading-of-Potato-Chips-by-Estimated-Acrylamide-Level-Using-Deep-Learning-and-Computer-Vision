
import pandas as pd
import matplotlib.pyplot as plt

input_file = "data/colour_analysis/brightness_by_grade_summary.csv"
output_file = "data/colour_analysis/mean_brightness_by_grade.png"

df = pd.read_csv(input_file)

# Keep the grades in the correct order
grade_order = ["Low", "Medium", "High", "Very High"]
df["grade"] = pd.Categorical(
    df["grade"], categories=grade_order, ordered=True
)
df = df.sort_values("grade")

plt.figure(figsize=(8, 5))
plt.bar(df["grade"].astype(str), df["mean_brightness"])
plt.xlabel("Measured Acrylamide Grade")
plt.ylabel("Mean Estimated Chip Brightness (0–255)")
plt.title("Mean Chip Brightness by Acrylamide Grade")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(output_file, dpi=300)
plt.show()

print("Graph saved to:", output_file)
print(df[["grade", "sample_count", "mean_brightness"]].to_string(index=False))
