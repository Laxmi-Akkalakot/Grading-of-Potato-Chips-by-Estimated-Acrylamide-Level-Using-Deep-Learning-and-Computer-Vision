
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(
    "models/image_validation/potato_chip_validator.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

image_path = input("Enter image path: ").strip().strip('"')

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not read image.")
    raise SystemExit(1)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = tf.keras.applications.mobilenet_v2.preprocess_input(
    image.astype(np.float32)
)

image = np.expand_dims(image, axis=0)

probability = float(model.predict(image, verbose=0)[0][0])

print("\n------ IMAGE VALIDATION RESULT ------")
print(f"Potato-chip probability: {probability * 100:.2f}%")

if probability >= 0.5:
    print("RESULT: Potato chips detected")
else:
    print("RESULT: Not potato chips")

print("-------------------------------------")
