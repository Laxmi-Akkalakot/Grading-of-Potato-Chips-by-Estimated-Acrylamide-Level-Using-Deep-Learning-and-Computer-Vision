
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(
    "models/image_validation/potato_chip_validator.keras"
)

TEST_DIR = Path(
    "data/image_validation/real_world_test"
)

model = tf.keras.models.load_model(MODEL_PATH)

extensions = {".jpg", ".jpeg", ".png", ".webp"}

images = [
    p for p in TEST_DIR.iterdir()
    if p.suffix.lower() in extensions
]

if not images:
    print("No test images found.")
    print("Please add photographs to:", TEST_DIR.resolve())
    raise SystemExit(0)

accepted = 0
rejected = 0

print("\n========== REAL-WORLD IMAGE TEST ==========\n")

for path in images:
    image = cv2.imread(str(path))

    if image is None:
        print(path.name, "-> Could not read image")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image.astype(np.float32)
    )

    image = np.expand_dims(image, axis=0)

    probability = float(
        model.predict(image, verbose=0)[0][0]
    )

    status = (
        "ACCEPTED as potato chips"
        if probability >= 0.5
        else "REJECTED as non-chip"
    )

    if probability >= 0.5:
        accepted += 1
    else:
        rejected += 1

    print(
        f"{path.name}: {probability * 100:.2f}% -> {status}"
    )

print("\n========== SUMMARY ==========")
print("Total images:", len(images))
print("Accepted:", accepted)
print("Rejected:", rejected)
print("==============================")
