
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT / "data" / "colour_analysis"
    / "cnn_confusion_matrix.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "colour_analysis"
    / "cnn_confusion_matrix.png"
)

# Load the saved confusion matrix
cm_df = pd.read_csv(CSV_PATH, index_col=0)

# Plot the matrix
fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(cm_df.values, cmap="Blues")

ax.set_xticks(range(len(cm_df.columns)))
ax.set_xticklabels(cm_df.columns, rotation=30, ha="right")
ax.set_yticks(range(len(cm_df.index)))
ax.set_yticklabels(cm_df.index)

ax.set_xlabel("Predicted Grade")
ax.set_ylabel("Actual Grade")
ax.set_title("CNN Confusion Matrix")

# Display each count inside its cell
for i in range(len(cm_df.index)):
    for j in range(len(cm_df.columns)):
        value = cm_df.iloc[i, j]
        ax.text(j, i, str(value), ha="center", va="center")

fig.colorbar(image, ax=ax, label="Number of Images")
fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300)
plt.show()

print("Confusion matrix graph saved to:", OUTPUT_PATH)
print(cm_df)
