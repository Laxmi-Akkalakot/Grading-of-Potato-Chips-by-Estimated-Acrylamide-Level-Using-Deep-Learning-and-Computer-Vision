
import os
import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load existing data splits
train_df = pd.read_csv("data/regression_train_split.csv")
test_df = pd.read_csv("data/regression_test_split.csv")


def extract_features(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.resize(img, (160, 160))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    # 1. BGR, HSV and LAB channel statistics
    for image in [img, hsv, lab]:
        for channel in cv2.split(image):
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])

    # 2. Colour histograms
    for image in [img, hsv, lab]:
        for channel in cv2.split(image):
            hist = cv2.calcHist(
                [channel], [0], None, [8], [0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

    # 3. Grayscale statistics
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.median(gray)
    ])

    # 4. Texture using Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient = np.sqrt(sobel_x**2 + sobel_y**2)

    features.extend([
        np.mean(gradient),
        np.std(gradient)
    ])

    # 5. Surface texture using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    features.extend([
        np.mean(np.abs(laplacian)),
        np.std(laplacian)
    ])

    return features


print("Extracting V2 training features...")

X_train = np.array([
    extract_features(path)
    for path in train_df["image_path"]
])

print("Extracting V2 testing features...")

X_test = np.array([
    extract_features(path)
    for path in test_df["image_path"]
])

y_train = train_df["acrylamide_ug_per_kg"].values
y_test = test_df["acrylamide_ug_per_kg"].values

# Log-transform target to reduce the influence of extreme values
y_train_log = np.log1p(y_train)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=3,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest V2...")

model.fit(X_train, y_train_log)

# Predict and convert back to original units
predictions_log = model.predict(X_test)
predictions = np.expm1(predictions_log)

predictions = np.maximum(predictions, 0)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== RANDOM FOREST V2 RESULTS ==========")
print(f"MAE:  {mae:.2f} µg/kg")
print(f"RMSE: {rmse:.2f} µg/kg")
print(f"R²:   {r2:.4f}")


# Convert predicted numerical values into grades
def assign_grade(value):

    if value <= 500:
        return "Low"
    elif value <= 1000:
        return "Medium"
    elif value <= 2000:
        return "High"
    else:
        return "Very High"


# Save predictions
results = test_df[
    ["image_name", "acrylamide_ug_per_kg", "grade"]
].copy()

results["predicted_acrylamide_ug_per_kg"] = predictions

results["predicted_grade"] = [
    assign_grade(value) for value in predictions
]

results["absolute_error"] = np.abs(
    results["acrylamide_ug_per_kg"] - predictions
)

os.makedirs("evaluation/regression", exist_ok=True)

results.to_csv(
    "evaluation/regression/random_forest_v2_predictions.csv",
    index=False
)

print("\nV2 predictions saved successfully!")

print("\n========== GRADE COMPARISON ==========")

print(
    pd.crosstab(
        results["grade"],
        results["predicted_grade"],
        rownames=["Actual Grade"],
        colnames=["Predicted Grade"]
    )
)
