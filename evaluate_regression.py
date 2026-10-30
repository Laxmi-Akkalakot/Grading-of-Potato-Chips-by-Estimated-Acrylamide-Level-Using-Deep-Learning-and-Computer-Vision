
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load test data and model predictions
df = pd.read_csv(
    "evaluation/regression_test_predictions.csv"
)

actual = df["acrylamide_ug_per_kg"].values
predicted = df["predicted_acrylamide_ug_per_kg"].values

# Baseline: always predict training-set median
train_df = pd.read_csv("data/regression_train_split.csv")
baseline_value = train_df["acrylamide_ug_per_kg"].median()

baseline_predictions = np.full(
    len(actual),
    baseline_value
)

# Evaluation function
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE:  {mae:.2f} µg/kg")
    print(f"RMSE: {rmse:.2f} µg/kg")
    print(f"R²:   {r2:.4f}")

# Compare both models
evaluate("MobileNetV2 Regression", actual, predicted)

evaluate(
    "Median Baseline",
    actual,
    baseline_predictions
)

# Show sample predictions
print("\nSample predictions:")
print(
    df[
        [
            "image_name",
            "acrylamide_ug_per_kg",
            "predicted_acrylamide_ug_per_kg",
            "absolute_error"
        ]
    ].head(10).to_string(index=False)
)
