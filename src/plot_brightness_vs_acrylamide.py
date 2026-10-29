
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load measured colour data
df = pd.read_csv(
    "data/colour_analysis/chip_colour_measurements.csv"
)

x = "estimated_chip_brightness"
y = "acrylamide_ug_per_kg"

df = df[[x, y]].dropna()

rho, p_value = spearmanr(df[x], df[y])

# Create scatter plot
plt.figure(figsize=(9, 6))
plt.scatter(df[x], df[y], alpha=0.65)
plt.xlabel("Estimated Chip Brightness (0–255)")
plt.ylabel("Laboratory Acrylamide (µg/kg)")
plt.title(
    f"Chip Brightness vs Acrylamide\n"
    f"Spearman correlation = {rho:.3f}"
)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output = (
    "data/colour_analysis/"
    "brightness_vs_acrylamide_correlation.png"
)
plt.savefig(output, dpi=300)
plt.show()

print("Spearman correlation:", round(rho, 4))
print("Graph saved to:", output)
