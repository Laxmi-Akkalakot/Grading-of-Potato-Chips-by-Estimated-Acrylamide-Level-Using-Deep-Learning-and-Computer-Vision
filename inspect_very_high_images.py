
import pandas as pd
import os

df = pd.read_csv("data/regression_train_split.csv")

very_high = df[df["acrylamide_ug_per_kg"] > 2000].copy()

print("\n========== VERY HIGH TRAINING IMAGES ==========")

print("Total Very High images:", len(very_high))

for _, row in very_high.iterrows():
    print(
        f"Image: {row['image_name']} | "
        f"Acrylamide: {row['acrylamide_ug_per_kg']:.2f} µg/kg | "
        f"Exists: {os.path.exists(row['image_path'])}"
    )

print("\n========== VALUE STATISTICS ==========")

print(
    very_high["acrylamide_ug_per_kg"].describe()
)
