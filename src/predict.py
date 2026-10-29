
from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf

# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "acrylamide_grade_checkpoint.keras"
CLASSES_PATH = ROOT / "models" / "grade_classes.json"

IMAGE_SIZE = (160, 160)

def main():
    # Ask for an image path if it wasn't supplied in the command.
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
    else:
        entered_path = input(
            "Enter potato-chip image path: "
        ).strip().strip('"')
        image_path = Path(entered_path)

        if not image_path.is_absolute():
            image_path = ROOT / image_path

    if not MODEL_PATH.exists():
        print("Trained model not found:", MODEL_PATH)
        return

    if not CLASSES_PATH.exists():
        print("Class names file not found:", CLASSES_PATH)
        return

    if not image_path.is_file():
        print("Image file not found:", image_path)
        return

    # Load model and class names.
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        classes = json.load(file)

    # Use the same image preprocessing as training.
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)

    # Predict grade.
    probabilities = model.predict(image, verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    predicted_grade = classes[best_index]
    confidence = float(probabilities[best_index]) * 100

    print("\n========== POTATO CHIP GRADING ==========")
    print("Image:", image_path.name)
    print("Predicted grade:", predicted_grade)
    print(f"Model confidence: {confidence:.2f}%")
    print("\nScores for each grade:")

    for name, score in sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{name}: {score * 100:.2f}%")

    print("\nNote: This is an experimental image-based estimate.")
    print("It does not measure acrylamide or establish food safety.")

if __name__ == "__main__":
    main()
