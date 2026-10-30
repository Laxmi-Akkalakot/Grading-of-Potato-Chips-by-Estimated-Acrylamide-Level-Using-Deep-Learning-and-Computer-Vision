
import os
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/classification/acrylamide_mobilenet_v2.keras"

CLASS_NAMES = ["Low", "Medium", "High", "Very High"]

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


def predict_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (160, 160))

    img = np.array(img, dtype=np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    probabilities = model.predict(img, verbose=0)[0]

    predicted_index = np.argmax(probabilities)

    predicted_grade = CLASS_NAMES[predicted_index]

    print("\n========== MOBILENETV2 PREDICTION ==========")

    print("Image:", os.path.basename(image_path))
    print("Predicted Grade:", predicted_grade)

    print("\nClass Scores:")

    for grade, probability in zip(
        CLASS_NAMES, probabilities
    ):
        print(f"{grade}: {probability * 100:.2f}%")

    print("\n============================================")


if __name__ == "__main__":

    image_path = input(
        "Enter potato chip image path: "
    ).strip().strip('"')

    predict_image(image_path)
