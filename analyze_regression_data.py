
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/labels.csv")

# Create output folder
os.makedirs("evaluation/regression", exist_ok=True)

# 1. Acrylamide value distribution
plt.figure(figsize=(10, 6))

plt.hist(
    df["acrylamide_ug_per_kg"],
    bins=30,
    edgecolor="black"
)

plt.title("Distribution of Acrylamide Values")
plt.xlabel("Acrylamide (µg/kg)")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.savefig(
    "evaluation/regression/acrylamide_distribution.png"
)
plt.show()


# 2. Grade-wise boxplot
grades = ["Low", "Medium", "High", "Very High"]

values = [
    df[df["grade"] == grade]["acrylamide_ug_per_kg"]
    for grade in grades
]

plt.figure(figsize=(10, 6))

plt.boxplot(values, tick_labels=grades)

plt.title("Acrylamide Values by Grade")
plt.xlabel("Grade")
plt.ylabel("Acrylamide (µg/kg)")

plt.tight_layout()
plt.savefig(
    "evaluation/regression/grade_wise_distribution.png"
)
plt.show()


# 3. Print statistics
print("\nGrade-wise statistics:")

print(
    df.groupby("grade")["acrylamide_ug_per_kg"]
    .agg(["count", "min", "max", "mean", "median", "std"])
)

print("\nDistribution analysis completed.")
