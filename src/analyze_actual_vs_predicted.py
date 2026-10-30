
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "all_models_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 2. LOAD PREDICTION DATA
# ==================================================

print("\nLoading prediction results...")

df = pd.read_csv(INPUT_FILE)

print("Total test images:", len(df))

# ==================================================
# 3. DEFINE MODELS AND GRADES
# ==================================================

CLASS_NAMES = [
    "High",
    "Low",
    "Medium",
    "Very High"
]

MODEL_COLUMNS = {
    "Original CNN": "Original CNN_prediction",
    "MobileNetV2 V2": "MobileNetV2 V2_prediction",
    "Fine-tuned MobileNetV2 V3":
        "Fine-tuned MobileNetV2 V3_prediction"
}

# ==================================================
# 4. CHECK DATA
# ==================================================

required_columns = [
    "actual_grade"
] + list(MODEL_COLUMNS.values())

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}"
        )

df = df.dropna(
    subset=required_columns
).copy()

# Check grade values
for column in required_columns:

    invalid = (
        set(df[column].astype(str)) - set(CLASS_NAMES)
    )

    if invalid:
        raise ValueError(
            f"Unexpected grade values in {column}: {invalid}"
        )

print("Valid prediction records:", len(df))

# ==================================================
# 5. CREATE ACTUAL VS PREDICTED ANALYSIS
# ==================================================

analysis_df = df.copy()

summary_results = []
grade_results = []
misclassified_records = []

for model_name, prediction_column in MODEL_COLUMNS.items():

    print("\n===================================")
    print("Model:", model_name)
    print("===================================")

    actual = analysis_df["actual_grade"].astype(str)

    predicted = (
        analysis_df[prediction_column].astype(str)
    )

    correct = actual == predicted

    correct_count = int(correct.sum())

    incorrect_count = int((~correct).sum())

    total = len(analysis_df)

    accuracy = (correct_count / total) * 100

    print("Correct predictions:", correct_count)
    print("Incorrect predictions:", incorrect_count)
    print(f"Accuracy: {accuracy:.2f}%")

    # ------------------------------------------------
    # Save overall model results
    # ------------------------------------------------

    summary_results.append({
        "Model": model_name,
        "Correct Predictions": correct_count,
        "Incorrect Predictions": incorrect_count,
        "Total Images": total,
        "Accuracy (%)": round(accuracy, 2)
    })

    # ------------------------------------------------
    # Save individual prediction results
    # ------------------------------------------------

    analysis_df[
        model_name + "_correct"
    ] = correct

    # ------------------------------------------------
    # Calculate grade-wise accuracy
    # ------------------------------------------------

    for grade in CLASS_NAMES:

        grade_mask = actual == grade

        grade_total = int(grade_mask.sum())

        grade_correct = int(
            (correct & grade_mask).sum()
        )

        grade_incorrect = grade_total - grade_correct

        grade_accuracy = (
            (grade_correct / grade_total) * 100
            if grade_total > 0 else 0
        )

        grade_results.append({
            "Model": model_name,
            "Grade": grade,
            "Correct": grade_correct,
            "Incorrect": grade_incorrect,
            "Total": grade_total,
            "Accuracy (%)": round(grade_accuracy, 2)
        })

    # ------------------------------------------------
    # Collect misclassified images
    # ------------------------------------------------

    wrong_indices = analysis_df.index[~correct]

    for index in wrong_indices:

        row = analysis_df.loc[index]

        misclassified_records.append({
            "Model": model_name,
            "Image Name": row.get("image_name", ""),
            "Image Path": row.get(
                "full_path",
                row.get("image_path", "")
            ),
            "Actual Grade": row["actual_grade"],
            "Predicted Grade": row[prediction_column]
        })

# ==================================================
# 6. SAVE INDIVIDUAL PREDICTION RESULTS
# ==================================================

prediction_columns = [
    "image_name",
    "image_path",
    "full_path",
    "actual_grade",
    "Original CNN_prediction",
    "MobileNetV2 V2_prediction",
    "Fine-tuned MobileNetV2 V3_prediction"
]

available_columns = [
    column for column in prediction_columns
    if column in analysis_df.columns
]

analysis_df[available_columns].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted_all_models.csv"
    ),
    index=False
)

# ==================================================
# 7. SAVE MISCLASSIFIED IMAGES
# ==================================================

misclassified_df = pd.DataFrame(
    misclassified_records
)

misclassified_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "misclassified_images_all_models.csv"
    ),
    index=False
)

# ==================================================
# 8. SAVE MODEL CORRECTNESS SUMMARY
# ==================================================

summary_df = pd.DataFrame(summary_results)

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_correctness_summary.csv"
    ),
    index=False
)

print("\nMODEL CORRECTNESS SUMMARY")
print(summary_df.to_string(index=False))

# ==================================================
# 9. SAVE GRADE-WISE ACCURACY
# ==================================================

grade_df = pd.DataFrame(grade_results)

grade_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "grade_wise_accuracy.csv"
    ),
    index=False
)

print("\nGRADE-WISE ACCURACY")
print(grade_df.to_string(index=False))

# ==================================================
# 10. CREATE CORRECT VS INCORRECT GRAPH
# ==================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(summary_df))

correct_values = summary_df[
    "Correct Predictions"
].values

incorrect_values = summary_df[
    "Incorrect Predictions"
].values

plt.bar(
    x,
    correct_values,
    label="Correct Predictions"
)

plt.bar(
    x,
    incorrect_values,
    bottom=correct_values,
    label="Incorrect Predictions"
)

for i in range(len(summary_df)):

    plt.text(
        i,
        correct_values[i] / 2,
        str(correct_values[i]),
        ha="center",
        va="center",
        fontweight="bold"
    )

    plt.text(
        i,
        correct_values[i] + incorrect_values[i] / 2,
        str(incorrect_values[i]),
        ha="center",
        va="center",
        fontweight="bold"
    )

plt.title(
    "Correct vs Incorrect Predictions",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Deep Learning Models")

plt.ylabel("Number of Images")

plt.xticks(
    x,
    summary_df["Model"],
    rotation=10
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted_analysis.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==================================================
# 11. CREATE GRADE-WISE ACCURACY HEATMAP
# ==================================================

heatmap_data = grade_df.pivot(
    index="Model",
    columns="Grade",
    values="Accuracy (%)"
)

plt.figure(figsize=(10, 5))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    vmin=0,
    vmax=100,
    linewidths=0.5
)

plt.title(
    "Grade-wise Classification Accuracy",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Acrylamide Grade")

plt.ylabel("Deep Learning Model")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "grade_wise_accuracy_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==================================================
# 12. FINAL OUTPUT
# ==================================================

print("\n===================================")
print("ACTUAL VS PREDICTED ANALYSIS DONE")
print("===================================")

print("\nGenerated files:")

print("1. actual_vs_predicted_all_models.csv")
print("2. misclassified_images_all_models.csv")
print("3. grade_wise_accuracy.csv")
print("4. model_correctness_summary.csv")
print("5. actual_vs_predicted_analysis.png")
print("6. grade_wise_accuracy_heatmap.png")

print("\nOutput folder:", OUTPUT_DIR)
