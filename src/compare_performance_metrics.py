
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report
)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "all_models_predictions.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 2. LOAD PREDICTION DATA
# ==================================================

print("\nLoading model prediction results...")

df = pd.read_csv(INPUT_FILE)

print("Total test images:", len(df))

# ==================================================
# 3. DEFINE CLASSES AND MODELS
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
    "Fine-tuned MobileNetV2 V3": "Fine-tuned MobileNetV2 V3_prediction"
}

# ==================================================
# 4. CHECK REQUIRED COLUMNS
# ==================================================

required_columns = ["actual_grade"] + list(MODEL_COLUMNS.values())

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column not found: {column}"
        )

# Remove rows with missing labels or predictions
df = df.dropna(subset=required_columns).copy()

# Validate class labels
for column in required_columns:
    invalid = set(df[column].astype(str)) - set(CLASS_NAMES)

    if invalid:
        raise ValueError(
            f"Unexpected labels in {column}: {invalid}"
        )

y_true = df["actual_grade"].astype(str)

# ==================================================
# 5. CALCULATE PRECISION, RECALL AND F1-SCORE
# ==================================================

summary_results = []
class_results = []

print("\n========================================")
print("MODEL PERFORMANCE METRICS")
print("========================================")

for model_name, prediction_column in MODEL_COLUMNS.items():

    print("\nEvaluating:", model_name)

    y_pred = df[prediction_column].astype(str)

    # Macro average metrics
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_NAMES,
            average="macro",
            zero_division=0
        )
    )

    # Weighted average metrics
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_NAMES,
            average="weighted",
            zero_division=0
        )
    )

    # Print macro results
    print(f"Macro Precision: {macro_precision * 100:.2f}%")
    print(f"Macro Recall:    {macro_recall * 100:.2f}%")
    print(f"Macro F1-score:  {macro_f1 * 100:.2f}%")

    # Save summary results
    summary_results.append({
        "Model": model_name,
        "Macro Precision (%)": round(macro_precision * 100, 2),
        "Macro Recall (%)": round(macro_recall * 100, 2),
        "Macro F1-score (%)": round(macro_f1 * 100, 2),
        "Weighted Precision (%)": round(weighted_precision * 100, 2),
        "Weighted Recall (%)": round(weighted_recall * 100, 2),
        "Weighted F1-score (%)": round(weighted_f1 * 100, 2)
    })

    # Calculate individual class metrics
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_NAMES,
            average=None,
            zero_division=0
        )
    )

    for i, class_name in enumerate(CLASS_NAMES):

        class_results.append({
            "Model": model_name,
            "Grade": class_name,
            "Precision (%)": round(precision[i] * 100, 2),
            "Recall (%)": round(recall[i] * 100, 2),
            "F1-score (%)": round(f1[i] * 100, 2),
            "Support": int(support[i])
        })

# ==================================================
# 6. CREATE SUMMARY DATAFRAME
# ==================================================

summary_df = pd.DataFrame(summary_results)

print("\n========================================")
print("FINAL PERFORMANCE COMPARISON")
print("========================================")

print(summary_df.to_string(index=False))

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "performance_metrics_comparison.csv"
    ),
    index=False
)

# ==================================================
# 7. SAVE CLASS-WISE METRICS
# ==================================================

class_df = pd.DataFrame(class_results)

class_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_class_metrics_all_models.csv"
    ),
    index=False
)

print("\nClass-wise metrics:")
print(class_df.to_string(index=False))

# ==================================================
# 8. GENERATE PRECISION, RECALL AND F1 GRAPH
# ==================================================

metrics = [
    "Macro Precision (%)",
    "Macro Recall (%)",
    "Macro F1-score (%)"
]

x = np.arange(len(summary_df))

width = 0.23

plt.figure(figsize=(12, 7))

for i, metric in enumerate(metrics):

    values = summary_df[metric].values

    plt.bar(
        x + (i - 1) * width,
        values,
        width,
        label=metric.replace("Macro ", "").replace(" (%)", "")
    )

    for j, value in enumerate(values):

        plt.text(
            x[j] + (i - 1) * width,
            value + 1,
            f"{value:.1f}",
            ha="center",
            fontsize=9
        )

plt.title(
    "Precision, Recall and F1-score Comparison",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Deep Learning Models", fontsize=12)

plt.ylabel("Performance (%)", fontsize=12)

plt.xticks(
    x,
    summary_df["Model"],
    rotation=10
)

plt.ylim(0, 110)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

graph_path = os.path.join(
    OUTPUT_DIR,
    "precision_recall_f1_comparison.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==================================================
# 9. GENERATE DETAILED CLASSIFICATION REPORTS
# ==================================================

for model_name, prediction_column in MODEL_COLUMNS.items():

    report = classification_report(
        y_true,
        df[prediction_column].astype(str),
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nClassification Report:", model_name)
    print(report)

# ==================================================
# 10. FINAL OUTPUT
# ==================================================

print("\n========================================")
print("PERFORMANCE EVALUATION COMPLETED")
print("========================================")

print("\nGenerated files:")

print("1. precision_recall_f1_comparison.png")
print("2. performance_metrics_comparison.csv")
print("3. per_class_metrics_all_models.csv")

print("\nOutput folder:", OUTPUT_DIR)
