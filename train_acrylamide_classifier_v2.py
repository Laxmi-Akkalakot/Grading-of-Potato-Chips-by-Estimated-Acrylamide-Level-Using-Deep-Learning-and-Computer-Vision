
import os
import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_sample_weight
import joblib

# Load existing splits
train_df = pd.read_csv("data/regression_train_split.csv")
test_df = pd.read_csv("data/regression_test_split.csv")

# Assign grades from actual numerical values
def assign_grade(value):
    if value <= 500:
        return "Low"
    elif value <= 1000:
        return "Medium"
    elif value <= 2000:
        return "High"
    else:
        return "Very High"


train_df["grade"] = train_df["acrylamide_ug_per_kg"].apply(assign_grade)
test_df["grade"] = test_df["acrylamide_ug_per_kg"].apply(assign_grade)

# Image feature extraction
def extract_features(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(image_path)

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

    return features


print("Extracting training features...")

X_train = np.array([
    extract_features(path)
    for path in train_df["image_path"]
])

print("Extracting testing features...")

X_test = np.array([
    extract_features(path)
    for path in test_df["image_path"]
])

y_train = train_df["grade"].values
y_test = test_df["grade"].values

# Handle class imbalance using sample weights
weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# Train classifier
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

print("\nTraining direct grade classifier...")

model.fit(
    X_train,
    y_train,
    sample_weight=weights
)

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\n========== CLASSIFICATION RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["Low", "Medium", "High", "Very High"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["Low", "Medium", "High", "Very High"]
    )
)

# Save model and predictions
os.makedirs("models/classification", exist_ok=True)
os.makedirs("evaluation/classification_v2", exist_ok=True)

joblib.dump(
    model,
    "models/classification/acrylamide_grade_rf_v2.pkl"
)

results = test_df[
    ["image_name", "acrylamide_ug_per_kg", "grade"]
].copy()

results["predicted_grade"] = predictions

results.to_csv(
    "evaluation/classification_v2/predictions.csv",
    index=False
)

print("\nModel saved successfully!")
print("Predictions saved successfully!")
