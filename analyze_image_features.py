
import os
import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load datasets
train_df = pd.read_csv("data/regression_train_split.csv")
test_df = pd.read_csv("data/regression_test_split.csv")

# Extract image colour features
def extract_features(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(image_path)

    img = cv2.resize(img, (160, 160))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    # BGR channel statistics
    for channel in cv2.split(img):
        features.extend([
            np.mean(channel),
            np.std(channel)
        ])

    # HSV channel statistics
    for channel in cv2.split(hsv):
        features.extend([
            np.mean(channel),
            np.std(channel)
        ])

    # Brightness
    features.extend([
        np.mean(gray),
        np.std(gray)
    ])

    return features


print("Extracting training image features...")

X_train = np.array([
    extract_features(path)
    for path in train_df["image_path"]
])

print("Extracting testing image features...")

X_test = np.array([
    extract_features(path)
    for path in test_df["image_path"]
])

y_train = train_df["acrylamide_ug_per_kg"].values
y_test = test_df["acrylamide_ug_per_kg"].values

# Train Random Forest regression
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== RANDOM FOREST RESULTS ==========")
print(f"MAE:  {mae:.2f} µg/kg")
print(f"RMSE: {rmse:.2f} µg/kg")
print(f"R²:   {r2:.4f}")

# Save results
results = test_df[[
    "image_name",
    "acrylamide_ug_per_kg",
    "grade"
]].copy()

results["predicted_acrylamide_ug_per_kg"] = predictions

results["absolute_error"] = abs(
    results["acrylamide_ug_per_kg"] - predictions
)

os.makedirs("evaluation/regression", exist_ok=True)

results.to_csv(
    "evaluation/regression/random_forest_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
