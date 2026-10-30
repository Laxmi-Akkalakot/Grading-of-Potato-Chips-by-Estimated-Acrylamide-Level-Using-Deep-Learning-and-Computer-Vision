
import os
import pandas as pd
from datetime import datetime

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

OUTPUT_FILE = os.path.join(
    EVALUATION_DIR,
    "final_evaluation_summary.txt"
)

# ==================================================
# 2. HELPER FUNCTION
# ==================================================

def load_csv(filename):

    path = os.path.join(EVALUATION_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required evaluation file not found: {filename}"
        )

    return pd.read_csv(path)


def format_table(dataframe):

    return dataframe.to_string(
        index=False,
        justify="left"
    )


# ==================================================
# 3. LOAD EVALUATION RESULTS
# ==================================================

print("\nLoading evaluation results...")

accuracy_df = load_csv(
    "accuracy_comparison.csv"
)

metrics_df = load_csv(
    "performance_metrics_comparison.csv"
)

grade_df = load_csv(
    "grade_wise_accuracy.csv"
)

correctness_df = load_csv(
    "model_correctness_summary.csv"
)

confusion_df = load_csv(
    "confusion_matrix_v3.csv"
)

# ==================================================
# 4. VALIDATE DATA
# ==================================================

if len(accuracy_df) != 3:
    raise ValueError(
        "Expected accuracy results for exactly three models."
    )

if len(metrics_df) != 3:
    raise ValueError(
        "Expected performance metrics for three models."
    )

if len(grade_df) != 12:
    raise ValueError(
        "Expected 12 grade-wise results: 3 models x 4 grades."
    )

if len(correctness_df) != 3:
    raise ValueError(
        "Expected correctness results for three models."
    )

if len(confusion_df) != 4:
    raise ValueError(
        "Expected a 4 x 4 V3 confusion matrix."
    )

# ==================================================
# 5. FIND BEST MODEL
# ==================================================

best_model_row = accuracy_df.loc[
    accuracy_df["Accuracy (%)"].idxmax()
]

best_model = best_model_row["Model"]

best_accuracy = float(
    best_model_row["Accuracy (%)"]
)

total_test_images = int(
    accuracy_df["Total Test Images"].iloc[0]
)

# ==================================================
# 6. PREPARE REPORT SECTIONS
# ==================================================

project_title = (
    "GRADING OF POTATO CHIPS BY ESTIMATED "
    "ACRYLAMIDE LEVEL USING DEEP LEARNING "
    "AND COMPUTER VISION"
)

dataset_details = """
Dataset Information:
--------------------

Total original dataset images: 310
Training images: 217
Validation images: 46
Testing images: 47

Classification categories:
1. High
2. Low
3. Medium
4. Very High

The dataset contains four estimated acrylamide grade categories.
The model predicts a grade from the input potato-chip image.
It does not directly measure acrylamide concentration.
"""

# ==================================================
# 7. MODEL ACCURACY
# ==================================================

accuracy_section = """
MODEL ACCURACY COMPARISON
=========================

""" + format_table(accuracy_df)

# ==================================================
# 8. PRECISION, RECALL AND F1
# ==================================================

metrics_section = """
PRECISION, RECALL AND F1-SCORE
==============================

The following table contains macro-average and weighted-average
classification metrics for the three models.

""" + format_table(metrics_df)

# ==================================================
# 9. GRADE-WISE PERFORMANCE
# ==================================================

grade_section = """
GRADE-WISE CLASSIFICATION ACCURACY
==================================

""" + format_table(grade_df)

# ==================================================
# 10. CORRECT AND INCORRECT PREDICTIONS
# ==================================================

correctness_section = """
CORRECT AND INCORRECT PREDICTIONS
=================================

""" + format_table(correctness_df)

# ==================================================
# 11. V3 CONFUSION MATRIX
# ==================================================

confusion_section = """
FINE-TUNED MOBILENETV2 V3 CONFUSION MATRIX
==========================================

Rows represent actual grades.
Columns represent predicted grades.

""" + format_table(confusion_df)

# ==================================================
# 12. BEST MODEL SELECTION
# ==================================================

best_model_section = f"""
BEST MODEL SELECTION
====================

Selected Model: {best_model}

Test Accuracy: {best_accuracy:.2f}%

The selected model achieved the highest test accuracy
among the three evaluated models on the same test dataset.

"""

# ==================================================
# 13. LIMITATIONS
# ==================================================

limitations_section = """
LIMITATIONS
===========

1. The test dataset contains only 47 images.

2. The number of test images in each grade is unequal.

3. Very High has only four test images, so its performance
   estimate is particularly uncertain.

4. Model predictions depend on the quality and reliability
   of the assigned dataset labels.

5. The system estimates an acrylamide grade from image data.
   It does not directly measure laboratory acrylamide
   concentration.

6. Results on a larger independent dataset are needed
   to assess generalization more reliably.
"""

# ==================================================
# 14. FUTURE IMPROVEMENTS
# ==================================================

future_section = """
FUTURE IMPROVEMENTS
===================

1. Collect more original potato-chip images.

2. Increase the number of samples in the Very High category.

3. Verify dataset labels against reliable laboratory
   measurements wherever possible.

4. Experiment with additional transfer-learning models.

5. Improve image preprocessing and data quality.

6. Evaluate the selected model on a larger independent dataset.

7. Develop a user-friendly application for image-based
   estimated acrylamide grade prediction.
"""

# ==================================================
# 15. FINAL CONCLUSION
# ==================================================

conclusion_section = f"""
FINAL CONCLUSION
================

Three deep learning models were evaluated for classifying
potato-chip images into four estimated acrylamide grade
categories.

The original CNN achieved {accuracy_df.iloc[
    accuracy_df["Model"].tolist().index("Original CNN")
]["Accuracy (%)"]:.2f}% test accuracy.

MobileNetV2 V2 achieved {accuracy_df.iloc[
    accuracy_df["Model"].tolist().index("MobileNetV2 V2")
]["Accuracy (%)"]:.2f}% test accuracy.

Fine-tuned MobileNetV2 V3 achieved {best_accuracy:.2f}%
test accuracy.

Among the evaluated models, {best_model} achieved the
highest test accuracy and was selected as the best-performing
model.

The evaluation demonstrates the potential of transfer
learning and fine-tuning for image-based estimated
acrylamide grade classification.

Further testing with more independent images and reliable
reference labels is required before making broader claims
about model performance.
"""

# ==================================================
# 16. COMBINE FINAL REPORT
# ==================================================

report = f"""
{project_title}

FINAL MODEL EVALUATION SUMMARY
Generated on: {datetime.now().strftime("%d-%m-%Y %H:%M")}

============================================================

{dataset_details}

{accuracy_section}

{metrics_section}

{grade_section}

{correctness_section}

{confusion_section}

{best_model_section}

{limitations_section}

{future_section}

{conclusion_section}

============================================================
END OF EVALUATION REPORT
============================================================
"""

# ==================================================
# 17. SAVE REPORT
# ==================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)

print("\n========================================")
print("FINAL EVALUATION REPORT GENERATED")
print("========================================")

print("\nSelected Model:", best_model)
print("Test Accuracy:", best_accuracy, "%")

print("\nSaved report:")
print(OUTPUT_FILE)

print("\nEvaluation summary generation completed!")
