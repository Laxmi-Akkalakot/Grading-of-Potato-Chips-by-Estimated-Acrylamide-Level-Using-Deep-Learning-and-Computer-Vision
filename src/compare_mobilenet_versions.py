
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16

df = pd.read_csv(DATA_DIR / "labels.csv")

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

_, temp_df = train_test_split(
    df, test_size=0.30, random_state=42,
    stratify=df["label"]
)

_, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42,
    stratify=temp_df["label"]
)

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)

images = np.stack([
    load_image(p).numpy()
    for p in test_df["full_path"]
])

actual = test_df["label"].to_numpy()

models = [
    "acrylamide_grade_v2.keras",
    "acrylamide_grade_v3.keras"
]

results = []

for filename in models:
    model_path = MODEL_DIR / filename

    print("\nEvaluating:", filename)

    model = tf.keras.models.load_model(
        model_path, compile=False
    )

    probabilities = model.predict(images, verbose=0)
    predicted = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(actual, predicted) * 100

    print("Accuracy:", round(accuracy, 2), "%")
    print(classification_report(
        actual,
        predicted,
        labels=np.arange(4),
        target_names=encoder.classes_,
        zero_division=0
    ))

    results.append({
        "Model": filename,
        "Test Accuracy (%)": round(accuracy, 2)
    })

comparison = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(
    MODEL_DIR / "mobilenet_model_comparison.csv",
    index=False
)

print("\nComparison saved successfully.")
