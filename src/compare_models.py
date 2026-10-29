
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
df = pd.read_csv(DATA_DIR / "labels.csv")

# Resolve image paths using the same project structure.
def resolve_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        return str(path)

    fallback = DATA_DIR / "images" / Path(str(value)).name
    return str(fallback)

image_col = "image_path"
df["full_path"] = df[image_col].apply(resolve_path)
df = df[df["full_path"].apply(lambda p: Path(p).exists())].copy()
df = df[df["grade"].isin(["Low", "Medium", "High", "Very High"])].copy()

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["grade"])

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

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (160, 160))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_df["full_path"].astype(str).to_numpy(),
        test_df["label"].to_numpy(dtype=np.int32)
    )
)
test_ds = test_ds.map(load_image).batch(16)

actual = test_df["label"].to_numpy()
class_names = encoder.classes_

print("Test images:", len(test_df))
print("Class order:", list(class_names))

for filename in [
    "acrylamide_grade.keras",
    "acrylamide_grade_backup.keras",
    "acrylamide_grade_checkpoint.keras",
    "acrylamide_grade_final.keras",
]:
    path = MODEL_DIR / filename
    if not path.exists() or path.stat().st_size == 0:
        print(f"\nSkipping missing/empty model: {filename}")
        continue

    model = tf.keras.models.load_model(path, compile=False)
    probs = model.predict(test_ds, verbose=0)
    predicted = np.argmax(probs, axis=1)

    print(f"\n{filename}")
    print(f"Accuracy: {accuracy_score(actual, predicted) * 100:.2f}%")
    print("Predicted class counts:", np.bincount(
        predicted, minlength=len(class_names)
    ))
    print("Confusion matrix:")
    print(confusion_matrix(
        actual, predicted, labels=np.arange(len(class_names))
    ))
