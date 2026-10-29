
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
IMG_SIZE = (160, 160)
BATCH_SIZE = 16

df = pd.read_csv(ROOT / "data" / "labels.csv")
df["label"] = LabelEncoder().fit_transform(df["grade"])

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED,
    stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED,
    stratify=temp_df["label"]
)

def load_image(image_path, label):
    path = tf.strings.join([str(ROOT), "/", image_path])
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

test_ds = tf.data.Dataset.from_tensor_slices((
    test_df["image_path"].values,
    test_df["label"].values
))
test_ds = test_ds.map(load_image).batch(BATCH_SIZE)

model_path = ROOT / "models" / "acrylamide_grade_checkpoint.keras"
model = tf.keras.models.load_model(model_path)

y_true = test_df["label"].to_numpy()
probabilities = model.predict(test_ds, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

classes = ["High", "Low", "Medium", "Very High"]

print("Test images:", len(y_true))
print("Actual class counts:", np.bincount(y_true, minlength=4))
print("Predicted class counts:", np.bincount(y_pred, minlength=4))

print("\nClassification report:")
print(classification_report(
    y_true, y_pred,
    labels=[0, 1, 2, 3],
    target_names=classes,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]))

print("\nFirst 10 predictions:")
for actual, predicted, probs in zip(
    y_true[:10], y_pred[:10], probabilities[:10]
):
    print(
        f"Actual: {classes[actual]:10s} | "
        f"Predicted: {classes[predicted]:10s} | "
        f"Probabilities: {np.round(probs, 3)}"
    )
print("\n========== MISCLASSIFIED TEST IMAGES ==========")


image_paths = test_df["image_path"].tolist()

for i, (actual, predicted) in enumerate(zip(y_true, y_pred)):
    if actual != predicted:
        confidence = float(np.max(probabilities[i])) * 100

        print(f"Image: {image_paths[i]}")
        print(f"Actual grade: {classes[actual]}")
        print(f"Predicted grade: {classes[predicted]}")
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 40)
