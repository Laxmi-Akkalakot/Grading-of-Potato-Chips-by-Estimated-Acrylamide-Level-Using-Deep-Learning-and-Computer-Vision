
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_CSV = os.path.join(
    BASE_DIR, "data", "mobilenet_test_split.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "acrylamide_grade_v3.keras"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (160, 160)

CLASS_NAMES = ["High", "Low", "Medium", "Very High"]

# --------------------------------------------------
# 2. Load original saved test split
# --------------------------------------------------

test_df = pd.read_csv(TEST_CSV)

print("Original test dataset loaded")
print("Total test images:", len(test_df))

# Check required columns
required_columns = ["full_path", "label", "grade"]

for column in required_columns:
    if column not in test_df.columns:
        raise ValueError(f"Missing required column: {column}")

# Check image files
missing_images = test_df[
    ~test_df["full_path"].apply(os.path.exists)
]

if len(missing_images) > 0:
    raise FileNotFoundError(
        f"{len(missing_images)} test images are missing."
    )

# --------------------------------------------------
# 3. Prepare actual labels
# --------------------------------------------------

X_paths = test_df["full_path"].tolist()

y_true = test_df["label"].astype(int).to_numpy()

print("\nClass distribution:")
print(test_df["grade"].value_counts())

# --------------------------------------------------
# 4. Load and preprocess images
# --------------------------------------------------

images = []

for image_path in X_paths:

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = tf.keras.utils.img_to_array(image)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    images.append(image)

X_test = np.array(images, dtype=np.float32)

print("\nImage array shape:", X_test.shape)

# --------------------------------------------------
# 5. Load fine-tuned V3 model
# --------------------------------------------------

print("\nLoading V3 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("V3 model loaded successfully")

# --------------------------------------------------
# 6. Generate predictions
# --------------------------------------------------

print("\nPredicting test images...")

probabilities = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

# --------------------------------------------------
# 7. Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

print("\nV3 TEST ACCURACY:")
print(f"{accuracy * 100:.2f}%")

print("Correct predictions:", np.sum(y_true == y_pred))
print("Incorrect predictions:", np.sum(y_true != y_pred))
print("Total test images:", len(y_true))

# --------------------------------------------------
# 8. Generate confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3]
)

print("\nConfusion Matrix:")
print(cm)

# --------------------------------------------------
# 9. Visualize confusion matrix
# --------------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5,
    annot_kws={"size": 13}
)

plt.title(
    "Confusion Matrix - Fine-tuned MobileNetV2 (V3)",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Predicted Grade", fontsize=12)
plt.ylabel("Actual Grade", fontsize=12)

plt.tight_layout()

matrix_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix_v3.png"
)

plt.savefig(
    matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# 10. Save confusion matrix CSV
# --------------------------------------------------

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{name}" for name in CLASS_NAMES],
    columns=[f"Predicted_{name}" for name in CLASS_NAMES]
)

cm_df.to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix_v3.csv")
)

# --------------------------------------------------
# 11. Generate classification report
# --------------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    os.path.join(OUTPUT_DIR, "classification_report_v3.csv")
)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

# --------------------------------------------------
# 12. Save actual vs predicted results
# --------------------------------------------------

results_df = test_df.copy()

results_df["actual_grade"] = [
    CLASS_NAMES[i] for i in y_true
]

results_df["predicted_grade"] = [
    CLASS_NAMES[i] for i in y_pred
]

results_df["correct_prediction"] = (
    results_df["actual_grade"] ==
    results_df["predicted_grade"]
)

results_df["prediction_confidence"] = np.max(
    probabilities,
    axis=1
)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted_v3.csv"),
    index=False
)

# --------------------------------------------------
# 13. Final message
# --------------------------------------------------

print("\n----------------------------------")
print("V3 EVALUATION COMPLETED")
print("----------------------------------")

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nGenerated files:")
print("1. confusion_matrix_v3.png")
print("2. confusion_matrix_v3.csv")
print("3. classification_report_v3.csv")
print("4. actual_vs_predicted_v3.csv")

print("\nOutput folder:", OUTPUT_DIR)
