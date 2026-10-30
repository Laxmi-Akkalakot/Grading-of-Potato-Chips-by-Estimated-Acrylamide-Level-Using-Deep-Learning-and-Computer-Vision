
import cv2
import joblib
import numpy as np
import sys
import os

MODEL_PATH = "models/regression/random_forest_acrylamide.pkl"

# Load saved model
model = joblib.load(MODEL_PATH)


def extract_features(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    img = cv2.resize(img, (160, 160))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    for channel in cv2.split(img):
        features.extend([
            np.mean(channel),
            np.std(channel)
        ])

    for channel in cv2.split(hsv):
        features.extend([
            np.mean(channel),
            np.std(channel)
        ])

    features.extend([
        np.mean(gray),
        np.std(gray)
    ])

    return np.array(features).reshape(1, -1)


def predict_acrylamide(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    features = extract_features(image_path)

    prediction = model.predict(features)[0]

    prediction = max(0, prediction)

    print("\n========== PREDICTION ==========")
    print("Image:", image_path)
    print(
        f"Estimated acrylamide value: "
        f"{prediction:.2f} µg/kg"
    )

    if prediction < 500:
        grade = "Low"
    elif prediction < 1000:
        grade = "Medium"
    elif prediction < 2000:
        grade = "High"
    else:
        grade = "Very High"

    print("Estimated grade:", grade)
    print("\nNote: This is an experimental image-based estimate,")
    print("not a direct chemical measurement.")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python predict_acrylamide.py "
            "<image_path>"
        )
    else:
        predict_acrylamide(sys.argv[1])
