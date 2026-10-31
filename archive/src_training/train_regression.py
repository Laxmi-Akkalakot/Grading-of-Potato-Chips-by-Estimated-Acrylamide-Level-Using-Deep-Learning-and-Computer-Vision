"""Train a colour-feature Random Forest to estimate acrylamide (µg/kg)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _find_project_root() -> Path:
    """Walk upward from this file until the folder containing config.py."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.py").exists():
            return parent
    raise RuntimeError(
        "Could not find config.py above " + str(here) + ". "
        "Keep this file inside the PotatoGuard project folder."
    )


ROOT = _find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED, ensure_runtime_dirs  # noqa: E402
from src.safeguards import backup_if_exists  # noqa: E402

FEATURE_COLUMNS = [
    "mean_blue",
    "std_blue",
    "mean_green",
    "std_green",
    "mean_red",
    "std_red",
    "mean_hue",
    "std_hue",
    "mean_saturation",
    "std_saturation",
    "mean_value",
    "std_value",
    "brightness",
    "brightness_std",
]


def _metrics(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def main() -> Path:
    ensure_runtime_dirs()

    features_path = PROCESSED_DIR / "image_features.csv"
    if not features_path.exists():
        from src.feature_extraction import extract_dataset_features

        extract_dataset_features()

    features = pd.read_csv(_require(features_path))
    train_split = pd.read_csv(_require(DATA_DIR / "regression_train_split.csv"))
    val_split = pd.read_csv(_require(DATA_DIR / "regression_validation_split.csv"))
    test_split = pd.read_csv(_require(DATA_DIR / "regression_test_split.csv"))

    def merge(split_df: pd.DataFrame) -> pd.DataFrame:
        merged = split_df.merge(features, on="image_name", suffixes=("", "_feat"))
        if "acrylamide_ug_per_kg" not in merged.columns:
            raise ValueError("Missing laboratory acrylamide values.")
        merged = merged.dropna(subset=["acrylamide_ug_per_kg"] + FEATURE_COLUMNS)
        return merged

    train_df = merge(train_split)
    val_df = merge(val_split)
    test_df = merge(test_split)
    print("Regression rows (train, val, test):", len(train_df), len(val_df), len(test_df))

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError(
            "One of the splits is empty after merging. Check that image_name "
            "values match between the split CSVs and image_features.csv."
        )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["acrylamide_ug_per_kg"])

    val_pred = np.clip(model.predict(val_df[FEATURE_COLUMNS]), 0, None)
    test_pred = np.clip(model.predict(test_df[FEATURE_COLUMNS]), 0, None)
    val_metrics = _metrics(val_df["acrylamide_ug_per_kg"], val_pred)
    test_metrics = _metrics(test_df["acrylamide_ug_per_kg"], test_pred)
    print("Validation:", val_metrics)
    print("Test:", test_metrics)

    out_dir = MODELS_DIR / "regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "random_forest_features.joblib"
    backup_if_exists(out_path)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target": "acrylamide_ug_per_kg",
            "note": "Image-based estimate, not a laboratory measurement.",
        },
        out_path,
    )

    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pred_path = metrics_dir / "regression_test_predictions.csv"
    test_out = test_df[["image_name", "acrylamide_ug_per_kg"]].copy()
    test_out["predicted_acrylamide_ug_per_kg"] = test_pred
    test_out["absolute_error"] = (
        test_out["predicted_acrylamide_ug_per_kg"] - test_out["acrylamide_ug_per_kg"]
    ).abs()
    test_out.to_csv(pred_path, index=False)

    summary = {
        "validation": val_metrics,
        "test": test_metrics,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "disclaimer": "Estimated concentration from image features, not lab measurement.",
    }
    (metrics_dir / "regression_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Saved", out_path)
    return out_path


if __name__ == "__main__":
    main()