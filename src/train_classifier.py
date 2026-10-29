from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. PATHS AND SETTINGS
# ==========================================
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "labels.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "acrylamide_grade.keras"
CLASSES_PATH = MODEL_DIR / "grade_classes.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

# ==========================================
# 2. LOAD AND CHECK LABELS
# ==========================================
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Labels file not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Accept either "image_path" or "path" for the image location
if "image_path" in df.columns:
    path_col = "image_path"
elif "path" in df.columns:
    path_col = "path"
else:
    raise ValueError("CSV must contain an 'image_path' or 'path' column.")

if "grade" not in df.columns:
    raise ValueError("CSV must contain a 'grade' column.")

df["full_path"] = df[path_col].apply(
    lambda p: str(ROOT / str(p).replace("\\", "/"))
)

df = df[df["full_path"].apply(lambda p: Path(p).is_file())].copy()
df = df[df["grade"].isin(
    ["Low", "Medium", "High", "Very High"]
)].copy()

# --- Remove duplicate images (prevents the same image in train AND test) ---
grade_counts = df.groupby("full_path")["grade"].nunique()
conflicting = grade_counts[grade_counts > 1].index
if len(conflicting) > 0:
    print(f"WARNING: {len(conflicting)} image(s) have conflicting grades; removed.")
    df = df[~df["full_path"].isin(conflicting)].copy()

before = len(df)
df = df.drop_duplicates(subset="full_path").copy()
print(f"Removed {before - len(df)} duplicate image record(s).")

if len(df) < 20:
    raise ValueError(
        f"Only {len(df)} valid images found. Check data/labels.csv."
    )

print("Valid unique images:", len(df))
print("\nClass counts:")
print(df["grade"].value_counts())

# ==========================================
# 3. ENCODE GRADES
# ==========================================
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["grade"])

print("\nClass mapping:")
for index, name in enumerate(encoder.classes_):
    print(index, "=", name)

# ==========================================
# 4. SPLIT DATA (70 / 15 / 15)
# ==========================================
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label"]
)

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

# ==========================================
# 5. LOAD AND PREPROCESS IMAGES
# ==========================================
def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def make_dataset(frame, training=False):
    paths = frame["full_path"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            len(frame),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# ==========================================
# 6. BUILD CNN MODEL
# ==========================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10)
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(160, 160, 3)),
    data_augmentation,

    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.40),
    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 7. TRAIN MODEL
# ==========================================
MODEL_DIR.mkdir(parents=True, exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_DIR / "acrylamide_grade_checkpoint.keras"),
    monitor="val_loss",
    save_best_only=True
)
]
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights from training samples only
classes = np.unique(train_df["label"])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].to_numpy()
)

class_weights = {
    int(label): float(weight)
    for label, weight in zip(classes, weights)
}

print("\nClass weights:")
for label, weight in class_weights.items():
    print(encoder.classes_[label], ":", round(weight, 3))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

FINAL_MODEL_PATH = MODEL_DIR / "acrylamide_grade_final.keras"
model.save(FINAL_MODEL_PATH)

print("Final model saved to:", FINAL_MODEL_PATH)

with open(CLASSES_PATH, "w", encoding="utf-8") as file:
    json.dump(encoder.classes_.tolist(), file, indent=2)

print("\nModel saved to:", MODEL_PATH)
print("Class names saved to:", CLASSES_PATH)

# ==========================================
# 8. EVALUATE ON HELD-OUT TEST DATA
# ==========================================
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

true_labels = np.concatenate([labels.numpy() for _, labels in test_ds])

probabilities = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

all_labels = np.arange(len(encoder.classes_))

print("\nClassification report:")
print(classification_report(
    true_labels,
    predicted_labels,
    labels=all_labels,
    target_names=encoder.classes_,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(true_labels, predicted_labels, labels=all_labels))

# ==========================================
# 9. SAVE EVALUATION RESULTS
# ==========================================
REPORT_DIR = ROOT / "data" / "colour_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

report_dict = classification_report(
    true_labels,
    predicted_labels,
    labels=all_labels,
    target_names=encoder.classes_,
    zero_division=0,
    output_dict=True
)
report_path = REPORT_DIR / "cnn_classification_report.csv"
pd.DataFrame(report_dict).transpose().to_csv(report_path, index=True)

cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
cm_path = REPORT_DIR / "cnn_confusion_matrix.csv"
pd.DataFrame(
    cm, index=encoder.classes_, columns=encoder.classes_
).to_csv(cm_path, index=True)

accuracy_path = REPORT_DIR / "cnn_test_accuracy.txt"
with open(accuracy_path, "w", encoding="utf-8") as file:
    file.write(f"Test loss: {test_loss:.4f}\n")
    file.write(f"Test accuracy: {test_accuracy:.4f}\n")

print("\nEvaluation files saved:")
print(report_path)
print(cm_path)
print(accuracy_path)

print("\nTraining and evaluation complete!")