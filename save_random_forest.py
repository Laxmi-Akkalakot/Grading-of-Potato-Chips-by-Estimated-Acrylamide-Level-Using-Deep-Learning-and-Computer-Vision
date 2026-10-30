
import os
import joblib
import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# Load training data
train_df = pd.read_csv("data/regression_train_split.csv")

# Feature extraction
def extract_features(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(image_path)

    img = cv2.resize(img, (160, 160))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    for channel in cv2.split(img):
        features.extend([np.mean(channel), np.std(channel)])

    for channel in cv2.split(hsv):
        features.extend([np.mean(channel), np.std(channel)])

    features.extend([np.mean(gray), np.std(gray)])

    return features


print("Extracting training features...")

X_train = np.array([
    extract_features(path)
    for path in train_df["image_path"]
])

y_train = train_df["acrylamide_ug_per_kg"].values

# Train final Random Forest on training data
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)

print("Training Random Forest...")
model.fit(X_train, y_train)

# Save model
os.makedirs("models/regression", exist_ok=True)

model_path = "models/regression/random_forest_acrylamide.pkl"

joblib.dump(model, model_path)

print("\nRandom Forest model saved successfully!")
print("Location:", model_path)
