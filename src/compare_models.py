
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

IMG_SIZE = (160, 160)
BATCH_SIZE = 16

# Load dataset
df = pd.read_csv(DATA_DIR / "labels.csv")

print("CSV columns:", list(df.columns))

# Resolve image paths
def resolve_path(value):
    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    if path.exists():
        return str(path)

    fallback = DATA_DIR / "images" / Path(str(value)).name
    return str(fallback)

df["full_path"] = df["image_path"].apply(resolve_path)

df = df[
    df["full_path"].apply(lambda p: Path(p).exists())
].copy()

classes_expected = ["High", "Low", "Medium", "Very High"]

df = df[df["grade"].isin(classes_expected)].copy()

# Fixed class mapping
class_names = classes_expected
class_to_index = {
    name: index for index, name in enumerate(class_names)
}

df["label"] = df["grade"].map(class_to_index).astype(int)

print("\nClass distribution:")
print(df["grade"].value_counts())

# Split dataset
_, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

_, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# Image preprocessing
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_df["full_path"].astype(str).to_numpy(),
        test_df["label"].to_numpy(dtype=np.int32)
    )
)

test_ds = test_ds.map(load_image).batch(BATCH_SIZE)

actual = test_df["label"].to_numpy()

print("\nTest images:", len(test_df))
print("Class order:", class_names)

# Compare saved models
model_files = [
    "acrylamide_grade.keras",
    "acrylamide_grade_backup.keras",
    "acrylamide_grade_checkpoint.keras",
    "acrylamide_grade_final.keras",
]

for filename in model_files:

    path = MODEL_DIR / filename

    if not path.exists() or path.stat().st_size == 0:
        print(f"\nSkipping missing/empty model: {filename}")
        continue

    print(f"\nLoading model: {filename}")

    try:
        model = tf.keras.models.load_model(
            path, compile=False
        )

        probs = model.predict(test_ds, verbose=0)

        if probs.shape[-1] != len(class_names):
            print("Output class count mismatch:", probs.shape)
            continue

        predicted = np.argmax(probs, axis=1)

        print(f"\nModel: {filename}")
        print(
            f"Accuracy: {accuracy_score(actual, predicted) * 100:.2f}%"
        )

        print(
            "Predicted class counts:",
            np.bincount(predicted, minlength=4)
        )

        print("Confusion matrix:")
        print(
            confusion_matrix(
                actual, predicted, labels=np.arange(4)
            )
        )

    except Exception as e:
        print(f"Could not evaluate {filename}")
        print("Error:", str(e))

print("\nModel comparison completed.")
