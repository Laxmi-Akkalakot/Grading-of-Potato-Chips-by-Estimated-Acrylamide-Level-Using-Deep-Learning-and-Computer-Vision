
import pandas as pd
import os

# Load the complete regression dataset
df = pd.read_csv("data/regression_labels.csv")

# Existing classification split files
split_files = {
    "train": "data/mobilenet_train_split.csv",
    "validation": "data/mobilenet_validation_split.csv",
    "test": "data/mobilenet_test_split.csv"
}

# Check and create regression splits
for split_name, file_path in split_files.items():

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing split file: {file_path}")

    split_df = pd.read_csv(file_path)

    # Match images using image_name
    merged_df = split_df[["image_name"]].merge(
        df,
        on="image_name",
        how="left",
        validate="one_to_one"
    )

    if merged_df["acrylamide_ug_per_kg"].isna().any():
        raise ValueError(f"Missing regression labels in {split_name} split.")

    output_path = f"data/regression_{split_name}_split.csv"

    merged_df.to_csv(output_path, index=False)

    print(f"{split_name.capitalize()} split created:")
    print("Images:", len(merged_df))
    print("Saved:", output_path)
    print()

print("All regression splits created successfully!")
