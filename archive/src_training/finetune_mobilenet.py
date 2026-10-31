
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = DATA_DIR / "colour_analysis"

CSV_PATH = DATA_DIR / "labels.csv"
SOURCE_MODEL = MODEL_DIR / "acrylamide_grade_v2.keras"
NEW_MODEL = MODEL_DIR / "acrylamide_grade_v3.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 15

tf.keras.utils.set_random_seed(SEED)

# 1. Load dataset
df = pd.read_csv(CSV_PATH)

def resolve_path(value):
    p = Path(str(value).replace("\\", "/"))
    if not p.is_absolute():
        p = ROOT / p
    if p.is_file():
        return str(p)
    return str(DATA_DIR / "images" / Path(str(value)).name)

df["full_path"] = df["image_path"].apply(resolve_path)

df = df[df["full_path"].apply(lambda p: Path(p).is_file())].copy()
df = df[df["grade"].isin(
    ["Low", "Medium", "High", "Very High"]
)].copy()

conflicts = df.groupby("full_path")["grade"].nunique()
conflicts = conflicts[conflicts > 1].index
df = df[~df["full_path"].isin(conflicts)].copy()
df = df.drop_duplicates("full_path").copy()

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["grade"])

# 2. Same split as previous experiment
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED,
    stratify=temp_df["label"]
)

# 3. MobileNetV2 preprocessing
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def make_dataset(frame, training=False):
    ds = tf.data.Dataset.from_tensor_slices((
        frame["full_path"].astype(str).to_numpy(),
        frame["label"].to_numpy(dtype=np.int32)
    ))
    if training:
        ds = ds.shuffle(
            len(frame), seed=SEED, reshuffle_each_iteration=True
        )
    return ds.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df, True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# 4. Load v2 model
if not SOURCE_MODEL.exists():
    raise FileNotFoundError(SOURCE_MODEL)

model = tf.keras.models.load_model(
    SOURCE_MODEL, compile=False
)

# 5. Unfreeze only deeper MobileNetV2 layers
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name:
        base_model = layer
        break

if base_model is None:
    raise ValueError("MobileNetV2 base model not found.")

base_model.trainable = True

# Freeze earlier layers; fine-tune deeper layers only
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep all BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("Trainable base layers:",
      sum(layer.trainable for layer in base_model.layers))

# 6. Compile with a small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 7. Class weights
classes = np.unique(train_df["label"])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].to_numpy()
)
class_weights = {
    int(k): float(v) for k, v in zip(classes, weights)
}

# 8. Fine-tune and save a separate model
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(NEW_MODEL),
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.CSVLogger(
        str(MODEL_DIR / "finetune_history.csv")
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save(NEW_MODEL)

# 9. Evaluate on test data
best_model = tf.keras.models.load_model(
    NEW_MODEL, compile=False
)

probabilities = best_model.predict(test_ds, verbose=1)
actual = test_df["label"].to_numpy()
predicted = np.argmax(probabilities, axis=1)

class_names = encoder.classes_
labels = np.arange(len(class_names))

print("\nFine-tuned model accuracy:",
      round(np.mean(actual == predicted) * 100, 2), "%")

print("\nClassification report:")
print(classification_report(
    actual, predicted,
    labels=labels,
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(actual, predicted, labels=labels)
print("\nConfusion matrix:")
print(cm)

pd.DataFrame(
    cm, index=class_names, columns=class_names
).to_csv(REPORT_DIR / "mobilenet_v3_confusion_matrix.csv")

report = classification_report(
    actual, predicted,
    labels=labels,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    REPORT_DIR / "mobilenet_v3_classification_report.csv"
)

with open(MODEL_DIR / "grade_classes_v3.json", "w") as f:
    json.dump(class_names.tolist(), f, indent=2)

print("\nFine-tuning completed.")
print("New model:", NEW_MODEL)
