
import os
import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/regression_train_split.csv")

very_high = df[
    df["acrylamide_ug_per_kg"] > 2000
].copy()

cols = 4
rows = math.ceil(len(very_high) / cols)

plt.figure(figsize=(16, rows * 3))

for index, (_, row) in enumerate(very_high.iterrows()):

    image = cv2.imread(row["image_path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(rows, cols, index + 1)
    plt.imshow(image)

    plt.title(
        f"{row['image_name']}\n"
        f"{row['acrylamide_ug_per_kg']:.0f} µg/kg",
        fontsize=8
    )

    plt.axis("off")

plt.tight_layout()

os.makedirs("evaluation/colour_analysis", exist_ok=True)

plt.savefig(
    "evaluation/colour_analysis/very_high_contact_sheet.png",
    dpi=150
)

plt.show()

print("Contact sheet saved successfully!")
