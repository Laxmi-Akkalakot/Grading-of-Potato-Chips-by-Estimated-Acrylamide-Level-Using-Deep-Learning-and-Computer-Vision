
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

# Configuration
IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 20

CLASS_NAMES = ["Low", "Medium", "High", "Very High"]

# Load existing splits
train_df = pd.read_csv("data/regression_train_split.csv")
val_df = pd.read_csv("data/regression_validation_split.csv")
test_df = pd.read_csv("data/regression_test_split.csv")

# Generate grades from actual acrylamide values
def assign_grade(value):
    if value <= 500:
        return "Low"
    elif value <= 1000:
        return "Medium"
    elif value <= 2000:
        return "High"
    return "Very High"

for df in [train_df, val_df, test_df]:
    df["grade"] = df["acrylamide_ug_per_kg"].apply(assign_grade)

# Encode labels
class_to_index = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

for df in [train_df, val_df, test_df]:
    df["label"] = df["grade"].map(class_to_index)

# Check image files
for df in [train_df, val_df, test_df]:
    missing = [
        path for path in df["image_path"]
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing image: {missing[0]}"
        )

# Build image datasets
def create_dataset(df, training=False):

    paths = df["image_path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    def load_image(path, label):
        image = tf.io.read_file(path)

        image = tf.io.decode_image(
            image, channels=3, expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image, [IMAGE_SIZE, IMAGE_SIZE]
        )

        image = preprocess_input(image)

        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            len(df), seed=42
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(train_df, training=True)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)

# Calculate class weights from training data
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=train_df["label"].values
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(weights)
}

print("\nTraining class counts:")
print(train_df["grade"].value_counts())

print("\nClass weights:")
print(class_weights)

# Load MobileNetV2 backbone
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Build classifier
inputs = layers.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.05)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)

outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6
    )
]

print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save model
os.makedirs("models/classification", exist_ok=True)

model.save(
    "models/classification/acrylamide_mobilenet_v2.keras"
)

# Evaluate on test set
print("\nEvaluating test dataset...")

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

probabilities = model.predict(test_ds)
y_pred = np.argmax(probabilities, axis=1)

print("\n========== MOBILENETV2 RESULTS ==========")

print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3]
))

print("\nModel saved successfully!")
