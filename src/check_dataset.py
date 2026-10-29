
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "labels.csv")

print("Total CSV rows:", len(df))
print("\nGrade counts:")
print(df["grade"].value_counts())

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["grade"])

print("\nClass mapping:")
for index, name in enumerate(encoder.classes_):
    print(f"{index}: {name}")

print("\nEncoded label counts:")
print(df["label"].value_counts().sort_index())

print("\nDuplicate image paths:", df["image_path"].duplicated().sum())
print("\nMissing grade values:", df["grade"].isna().sum())
