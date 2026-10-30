
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = DATA_DIR / "colour_analysis"

CSV_PATH = DATA_DIR / "labels.csv"
MODEL_PATH = MODEL_DIR / "acrylamide_grade_v2.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

tf.keras.utils.set_random_seed(SEED)

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(CSV_PATH)

required = {"image_path", "grade"}
if not required.issubset(df.columns):
    raise ValueError("CSV must contain image_path and grade columns.")

def resolve_path(value):
    p = Path(str(value).replace("\\", "/"))

    if not p.is_absolute():
        p = ROOT / p

    if p.is_file():
        return str(p)

    return str(DATA_DIR / "images" / Path(str(value)).name)

df["full_path"] = df["image_path"].apply(resolve_path)

df = df[
    df["full_path"].apply(lambda p: Path(p).is_file())
].copy()

classes_expected = ["High", "Low", "Medium", "Very High"]
df = df[df["grade"].isin(classes_expected)].copy()

# Remove conflicting and duplicate image paths
conflicts = (
    df.groupby("full_path")["grade"].nunique()
)

conflicts = conflicts[conflicts > 1].index
df = df[~df["full_path"].isin(conflicts)].copy()
df = df.drop_duplicates(subset="full_path").copy()

if df.empty:
    raise ValueError("No valid images found.")

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["grade"])

print("Total images:", len(df))
print("\nClass distribution:")
print(df["grade"].value_counts())
print("\nClass mapping:", encoder.classes_.tolist())

# ==========================================
# 3. TRAIN / VALIDATION / TEST SPLIT
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

print("\nTraining:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

# Save exact split membership for reproducibility
train_df.assign(split="train").to_csv(
    DATA_DIR / "mobilenet_train_split.csv", index=False
)
val_df.assign(split="validation").to_csv(
    DATA_DIR / "mobilenet_validation_split.csv", index=False
)
test_df.assign(split="test").to_csv(
    DATA_DIR / "mobilenet_test_split.csv", index=False
)

# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    # MobileNetV2 expects values in [-1, 1]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


def make_dataset(frame, training=False):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            frame["full_path"].astype(str).to_numpy(),
            frame["label"].to_numpy(dtype=np.int32)
        )
    )

    if training:
        ds = ds.shuffle(
            len(frame),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# ==========================================
# 5. BUILD MOBILENETV2
# ==========================================

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10)
])

inputs = tf.keras.Input(shape=(160, 160, 3))

x = augmentation(inputs)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.30)(x)

outputs = tf.keras.layers.Dense(
    len(encoder.classes_),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 6. CLASS WEIGHTS
# ==========================================

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

print("\nClass weights:", class_weights)

# ==========================================
# 7. TRAIN MODEL
# ==========================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.CSVLogger(
        str(MODEL_DIR / "mobilenet_history.csv")
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save the best in-memory model
model.save(MODEL_PATH)

print("\nMobileNetV2 training completed.")
print("Model saved:", MODEL_PATH)

# Save class names
with open(
    MODEL_DIR / "grade_classes_v2.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(encoder.classes_.tolist(), f, indent=2)

# ==========================================
# 8. TEST EVALUATION
# ==========================================

best_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

probabilities = best_model.predict(
    test_ds,
    verbose=1
)

actual = test_df["label"].to_numpy()
predicted = np.argmax(probabilities, axis=1)

labels = np.arange(len(encoder.classes_))

print("\nClassification Report:")
print(classification_report(
    actual,
    predicted,
    labels=labels,
    target_names=encoder.classes_,
    zero_division=0
))

cm = confusion_matrix(
    actual,
    predicted,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)

pd.DataFrame(
    cm,
    index=encoder.classes_,
    columns=encoder.classes_
).to_csv(
    REPORT_DIR / "mobilenet_confusion_matrix.csv"
)

report = classification_report(
    actual,
    predicted,
    labels=labels,
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    REPORT_DIR / "mobilenet_classification_report.csv"
)

print("\nEvaluation files saved in:", REPORT_DIR)
