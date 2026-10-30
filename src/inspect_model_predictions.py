
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

IMG_SIZE = (160, 160)
CLASSES = ["High", "Low", "Medium", "Very High"]

df = pd.read_csv(DATA_DIR / "labels.csv")

df["full_path"] = df["image_path"].apply(
    lambda x: str(ROOT / x) if (ROOT / x).exists()
    else str(DATA_DIR / "images" / Path(x).name)
)

df = df[df["full_path"].apply(lambda x: Path(x).exists())].copy()
df = df[df["grade"].isin(CLASSES)].copy()

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
    image = tf.image.resize(image, IMG_SIZE)
    return tf.cast(image, tf.float32) / 255.0

images = np.stack([
    load_image(p).numpy()
    for p in test_df["full_path"]
])

print("Test images:", len(images))
print("Class order:", CLASSES)

models = [
    "acrylamide_grade.keras",
    "acrylamide_grade_backup.keras",
    "acrylamide_grade_final.keras"
]

for filename in models:
    path = MODEL_DIR / filename

    print("\n------------------------------")
    print("MODEL:", filename)

    model = tf.keras.models.load_model(path, compile=False)
    probabilities = model.predict(images, verbose=0)

    print("Output shape:", probabilities.shape)
    print("First 5 probability predictions:")
    print(np.round(probabilities[:5], 4))

    print("Average probability per class:")
    print(np.round(probabilities.mean(axis=0), 4))

    print("Prediction counts:")
    print(np.bincount(
        np.argmax(probabilities, axis=1),
        minlength=4
    ))
