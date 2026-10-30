
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_CSV = os.path.join(
    BASE_DIR, "data", "mobilenet_test_split.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (160, 160)

CLASS_NAMES = ["High", "Low", "Medium", "Very High"]

# ==================================================
# 2. DEFINE THE THREE MODELS
# ==================================================

MODEL_FILES = {
    "Original CNN": "acrylamide_grade.keras",
    "MobileNetV2 V2": "acrylamide_grade_v2.keras",
    "Fine-tuned MobileNetV2 V3": "acrylamide_grade_v3.keras"
}

# ==================================================
# 3. LOAD ORIGINAL TEST DATASET
# ==================================================

print("\nLoading original test dataset...")

test_df = pd.read_csv(TEST_CSV)

print("Total test images:", len(test_df))

if len(test_df) != 47:
    raise ValueError(
        f"Expected 47 test images, but found {len(test_df)}."
    )

required_columns = ["full_path", "label", "grade"]

for column in required_columns:
    if column not in test_df.columns:
        raise ValueError(f"Missing column: {column}")

# Verify that all image files exist
missing_images = test_df[
    ~test_df["full_path"].apply(os.path.exists)
]

if len(missing_images) > 0:
    raise FileNotFoundError(
        f"{len(missing_images)} test images are missing."
    )

# Actual class labels
y_true = test_df["label"].astype(int).to_numpy()

# ==================================================
# 4. LOAD AND PREPROCESS TEST IMAGES
# ==================================================

print("\nLoading test images...")

images = []

for image_path in test_df["full_path"]:

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = tf.keras.utils.img_to_array(image)

    images.append(image)

X_raw = np.array(images, dtype=np.float32)

print("Image dataset shape:", X_raw.shape)

# Original CNN preprocessing: values between 0 and 1
X_cnn = X_raw / 255.0

# MobileNetV2 preprocessing: approximately -1 to +1
X_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(
    X_raw.copy()
)

# ==================================================
# 5. EVALUATE EACH MODEL
# ==================================================

results = []

prediction_results = test_df.copy()

prediction_results["actual_grade"] = [
    CLASS_NAMES[i] for i in y_true
]

for model_name, model_file in MODEL_FILES.items():

    print("\n======================================")
    print("Evaluating:", model_name)
    print("======================================")

    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    # Load saved model
    model = tf.keras.models.load_model(model_path)

    # Select correct preprocessing
    if model_name == "Original CNN":
        X_input = X_cnn
    else:
        X_input = X_mobilenet

    # Check output classes
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"{model_name} does not have four output classes."
        )

    # Generate predictions
    probabilities = model.predict(
        X_input,
        verbose=0
    )

    y_pred = np.argmax(probabilities, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    correct = int(np.sum(y_true == y_pred))
    incorrect = int(np.sum(y_true != y_pred))

    print("Correct predictions:", correct)
    print("Incorrect predictions:", incorrect)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save result
    results.append({
        "Model": model_name,
        "Correct Predictions": correct,
        "Incorrect Predictions": incorrect,
        "Total Test Images": len(y_true),
        "Accuracy (%)": round(accuracy * 100, 2)
    })

    # Save each model's predictions
    prediction_results[
        model_name + "_prediction"
    ] = [
        CLASS_NAMES[i] for i in y_pred
    ]

# ==================================================
# 6. CREATE ACCURACY COMPARISON TABLE
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy (%)",
    ascending=False
)

print("\n======================================")
print("FINAL MODEL ACCURACY COMPARISON")
print("======================================")

print(results_df.to_string(index=False))

# Save comparison table
results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_comparison.csv"
    ),
    index=False
)

# ==================================================
# 7. CREATE ACCURACY COMPARISON GRAPH
# ==================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    results_df["Model"],
    results_df["Accuracy (%)"],
    width=0.55
)

# Display accuracy above each bar
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.title(
    "Accuracy Comparison of Deep Learning Models",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Deep Learning Models", fontsize=12)

plt.ylabel("Test Accuracy (%)", fontsize=12)

plt.ylim(0, 100)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.xticks(rotation=10)

plt.tight_layout()

# Save graph
graph_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_comparison.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==================================================
# 8. SAVE INDIVIDUAL PREDICTIONS
# ==================================================

prediction_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_models_predictions.csv"
    ),
    index=False
)

# ==================================================
# 9. FINAL RESULT
# ==================================================

best_model = results_df.iloc[0]

print("\n======================================")
print("BEST PERFORMING MODEL")
print("======================================")

print("Model:", best_model["Model"])
print("Accuracy:", best_model["Accuracy (%)"], "%")

print("\nGenerated files:")
print("1. accuracy_comparison.png")
print("2. accuracy_comparison.csv")
print("3. all_models_predictions.csv")

print("\nEvaluation completed successfully!")
print("Output folder:", OUTPUT_DIR)
