
from pathlib import Path
import json
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent

POSITIVE_DIR = ROOT / "data/image_validation/potato_chips"
NEGATIVE_DIR = ROOT / "data/image_validation/not_potato_chips"

MODEL_DIR = ROOT / "models/image_validation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------- LOAD IMAGE PATHS ----------------

positive_images = [
    str(p) for p in POSITIVE_DIR.iterdir()
    if p.suffix.lower() in EXTENSIONS
]

negative_images = [
    str(p) for p in NEGATIVE_DIR.iterdir()
    if p.suffix.lower() in EXTENSIONS
]

if len(positive_images) < 20 or len(negative_images) < 20:
    raise ValueError("At least 20 images per class are required.")

# Balance the classes.
# This prevents the larger negative class dominating training.
random.shuffle(negative_images)
negative_images = negative_images[:len(positive_images)]

paths = positive_images + negative_images
labels = [1] * len(positive_images) + [0] * len(negative_images)

print("Potato-chip images:", len(positive_images))
print("Balanced non-chip images:", len(negative_images))
print("Total images:", len(paths))

# ---------------- SPLIT DATA ----------------

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)

print("Training:", len(train_paths))
print("Validation:", len(val_paths))
print("Testing:", len(test_paths))

# ---------------- IMAGE LOADING ----------------

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_paths, train_labels, True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# ---------------- BUILD MODEL ----------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10)
])

inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# ---------------- TRAIN ----------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

print("\nStarting improved validator training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------- EVALUATE ----------------

print("\nEvaluating test dataset...")

results = model.evaluate(test_ds, return_dict=True)

predictions = model.predict(test_ds, verbose=0).ravel()
predicted_labels = (predictions >= 0.5).astype(int)

print("\nClassification Report:")

report = classification_report(
    test_labels,
    predicted_labels,
    labels=[0, 1],
    target_names=["Not Potato Chips", "Potato Chips"],
    zero_division=0,
    output_dict=True
)

print(classification_report(
    test_labels,
    predicted_labels,
    labels=[0, 1],
    target_names=["Not Potato Chips", "Potato Chips"],
    zero_division=0
))

matrix = confusion_matrix(
    test_labels,
    predicted_labels,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(matrix)

# ---------------- SAVE MODEL ----------------

model_path = MODEL_DIR / "potato_chip_validator.keras"

model.save(model_path)

metrics = {
    "test_metrics": {
        key: float(value)
        for key, value in results.items()
    },
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "training_images": len(train_paths),
    "validation_images": len(val_paths),
    "test_images": len(test_paths),
    "threshold": 0.5,
    "positive_class": "potato_chips",
    "negative_class": "not_potato_chips",
    "warning": "Random split; near-duplicate images may inflate metrics."
}

with open(MODEL_DIR / "validation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nTraining completed!")
print("Model saved:", model_path)
print("Metrics saved:", MODEL_DIR / "validation_metrics.json")
