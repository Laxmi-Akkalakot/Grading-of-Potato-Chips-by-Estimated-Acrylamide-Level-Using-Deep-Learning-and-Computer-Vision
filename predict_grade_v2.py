
import os
import cv2
import numpy as np
import joblib

# Load trained classification model
MODEL_PATH = "models/classification/acrylamide_grade_rf_v2.pkl"

model = joblib.load(MODEL_PATH)

def extract_features(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    img = cv2.resize(img, (160, 160))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    # Colour statistics
    for image in [img, hsv, lab]:
        for channel in cv2.split(image):
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])

    # Colour histograms
    for image in [img, hsv, lab]:
        for channel in cv2.split(image):
            hist = cv2.calcHist(
                [channel], [0], None, [16], [0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

    # Grayscale statistics
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.median(gray)
    ])

    # Texture features
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    gradient = np.sqrt(sobel_x**2 + sobel_y**2)

    features.extend([
        np.mean(gradient),
        np.std(gradient)
    ])

    return np.array(features).reshape(1, -1)


def predict_grade(image_path):

    features = extract_features(image_path)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    print("\n========== ACRYLAMIDE PREDICTION ==========")

    print("Image:", os.path.basename(image_path))
    print("Predicted Grade:", prediction)

    print("\nClass Probabilities:")

    for grade, probability in zip(
        model.classes_, probabilities
    ):
        print(f"{grade}: {probability * 100:.2f}%")

    print("\n===========================================")


if __name__ == "__main__":

    image_path = input(
        "Enter potato chip image path: "
    ).strip().strip('"')

    predict_grade(image_path)
