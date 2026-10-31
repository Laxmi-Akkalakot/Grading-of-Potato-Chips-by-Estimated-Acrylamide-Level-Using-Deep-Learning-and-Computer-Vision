from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "acrylamide_grade.keras"
LABELS_PATH = DATA_DIR / "labels.csv"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42

df = pd.read_csv(LABELS_PATH)
print("CSV columns:", df.columns.tolist())

if "grade" not in df.columns:
    raise ValueError("Expected a 'grade' column in labels.csv")

image_col = next(
    (c for c in ["image_path", "path", "image", "filename", "image_name"]
     if c in df.columns),
    None
)
if image_col is None:
    raise ValueError("Cannot find image column. Check the CSV columns above.")

def resolve_image_path(value):
    path = Path(str(value).replace("\\", "/"))
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        return path
    return DATA_DIR / "images" / Path(str(value).replace("\\", "/")).name

df["resolved_path"] = df[image_col].apply(resolve_image_path)
df = df[df["resolved_path"].apply(lambda p: p.exists())].copy()
df = df[df["grade"].isin(["Low", "Medium", "High", "Very High"])].copy()

# Same duplicate handling as the training script, so the split matches.
df["key"] = df["resolved_path"].astype(str)
conflicting = df.groupby("key")["grade"].nunique()
conflicting = conflicting[conflicting > 1].index
df = df[~df["key"].isin(conflicting)].copy()
df = df.drop_duplicates(subset="key").copy()

if df.empty:
    raise ValueError("No valid image rows found. Check labels.csv paths.")

print("Unique valid images:", len(df))
print(df["grade"].value_counts())

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["grade"])

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_df["resolved_path"].astype(str).to_numpy(),
     test_df["label"].to_numpy(dtype=np.int32))
)
test_ds = (
    test_ds
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# This script only evaluates; it does not train or save a model.
model = tf.keras.models.load_model(MODEL_PATH)

loss, accuracy = model.evaluate(test_ds, verbose=1)
probabilities = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)
actual_labels = test_df["label"].to_numpy()
class_names = encoder.classes_
all_labels = np.arange(len(class_names))

print("\nModel:", MODEL_PATH.name)
print("Test images:", len(test_df))
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("Predicted class counts:", np.bincount(predicted_labels, minlength=len(class_names)))
print("Actual class counts:   ", np.bincount(actual_labels, minlength=len(class_names)))

print("\nClassification report:")
print(classification_report(
    actual_labels, predicted_labels,
    labels=all_labels, target_names=class_names, zero_division=0
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(actual_labels, predicted_labels, labels=all_labels))

output_dir = DATA_DIR / "colour_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

report = classification_report(
    actual_labels, predicted_labels,
    labels=all_labels, target_names=class_names,
    output_dict=True, zero_division=0
)
pd.DataFrame(report).transpose().to_csv(
    output_dir / "cnn_restored_classification_report.csv"
)
pd.DataFrame(
    confusion_matrix(actual_labels, predicted_labels, labels=all_labels),
    index=class_names, columns=class_names
).to_csv(output_dir / "cnn_restored_confusion_matrix.csv")

with open(output_dir / "cnn_restored_test_accuracy.txt", "w") as f:
    f.write(f"Test accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Test loss: {loss:.4f}\n")
    f.write(f"Test images: {len(test_df)}\n")

print("\nEvaluation files saved in:", output_dir)