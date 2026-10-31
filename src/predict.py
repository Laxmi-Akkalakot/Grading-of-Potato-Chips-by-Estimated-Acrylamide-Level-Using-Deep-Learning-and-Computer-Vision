
"""Load the acrylamide regression model and predict a single potato-chip image."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from config import (
    DISCLAIMER,
    LEGACY_REGRESSION_PATH,
    MODELS_DIR,
    grade_from_acrylamide,
)
from src.feature_extraction import (
    extract_features_from_bgr,
    regression_vector_from_features,
)
from src.safeguards import ModelReliabilityError


def extract_legacy_rf_features(bgr: np.ndarray) -> np.ndarray:
    """
    Extract the 14 full-image features used by the legacy Random Forest.
    Feature order must match save_random_forest.py.
    """
    image = cv2.resize(bgr, (160, 160))

    blue, green, red = cv2.split(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = [
        np.mean(blue), np.std(blue),
        np.mean(green), np.std(green),
        np.mean(red), np.std(red),
        np.mean(hue), np.std(hue),
        np.mean(saturation), np.std(saturation),
        np.mean(value), np.std(value),
        np.mean(gray), np.std(gray),
    ]

    return np.array([features], dtype=np.float64)


class Predictor:
    """Predict estimated acrylamide concentration from a potato-chip image."""

    def __init__(self) -> None:
        self.regression_bundle = None
        self.model_version = "Unknown"
        self._load()

    def _load(self) -> None:
        """Load the available regression model without replacing existing files."""

        new_model_path = (
            MODELS_DIR / "regression" / "random_forest_features.joblib"
        )
        legacy_model_path = LEGACY_REGRESSION_PATH

        if new_model_path.exists():
            model_path = new_model_path
            is_legacy = False
        elif legacy_model_path.exists():
            model_path = legacy_model_path
            is_legacy = True
        else:
            raise ModelReliabilityError(
                "No regression model found. Expected either "
                f"{new_model_path} or {legacy_model_path}"
            )

        try:
            loaded = joblib.load(model_path)

            if isinstance(loaded, dict) and "model" in loaded:
                self.regression_bundle = loaded
                self.regression_bundle.setdefault("legacy", is_legacy)
            else:
                self.regression_bundle = {
                    "model": loaded,
                    "legacy": is_legacy,
                }

            self.model_version = model_path.name

        except Exception as exc:
            raise ModelReliabilityError(
                f"Could not load regression model: {exc}"
            ) from exc

    def predict_bgr(
        self,
        bgr: np.ndarray,
        filename: str = "upload.jpg",
    ) -> dict:
        """Predict the concentration and derive its grade."""

        if bgr is None or not isinstance(bgr, np.ndarray):
            raise ModelReliabilityError("Invalid image supplied.")

        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ModelReliabilityError(
                "The uploaded image must be a valid colour image."
            )

        height, width = bgr.shape[:2]

        if min(height, width) < 16:
            raise ModelReliabilityError(
                f"Invalid image dimensions: {width}x{height}"
            )

        # Extract image features and prepare the segmentation preview.
        features, mask, preview = extract_features_from_bgr(bgr)

        bundle = self.regression_bundle
        model = bundle["model"]

        # Use the exact feature representation expected by the model.
        columns = bundle.get("feature_columns")

        if columns:
            missing = [column for column in columns if column not in features]

            if missing:
                raise ModelReliabilityError(
                    f"Required regression features are missing: {missing}"
                )

            vector = np.array(
                [[features[column] for column in columns]],
                dtype=np.float64,
            )

        elif bundle.get("legacy", False):
            # Legacy RF was trained on full, unsegmented images.
            vector = extract_legacy_rf_features(bgr)

        else:
            vector = regression_vector_from_features(features)

        try:
            raw_prediction = model.predict(vector)[0]
            estimated_value = float(raw_prediction)

            if not np.isfinite(estimated_value):
                raise ValueError("Model returned a non-finite value.")

            estimated_value = max(0.0, estimated_value)

        except Exception as exc:
            raise ModelReliabilityError(
                f"Acrylamide prediction failed: {exc}"
            ) from exc

        # The grade is derived from the numerical estimate.
        grade = grade_from_acrylamide(estimated_value)

        # Encode the segmentation preview.
        ok, preview_buffer = cv2.imencode(".jpg", preview)

        preview_b64 = None

        if ok:
            preview_b64 = base64.b64encode(
                preview_buffer.tobytes()
            ).decode("ascii")

        return {
            "filename": filename,
            "grade": grade,
            "confidence_percent": None,
            "class_confidences": {},
            "estimated_acrylamide_ug_per_kg": round(
                estimated_value, 2
            ),
            "features": {
                "mean_red": round(float(features["mean_red"]), 2),
                "mean_green": round(float(features["mean_green"]), 2),
                "mean_blue": round(float(features["mean_blue"]), 2),
                "mean_hue": round(float(features["mean_hue"]), 2),
                "mean_saturation": round(
                    float(features["mean_saturation"]), 2
                ),
                "mean_value": round(float(features["mean_value"]), 2),
                "brightness": round(float(features["brightness"]), 2),
                "saturation": round(float(features["saturation"]), 2),
                "foreground_fraction": round(
                    float(features["foreground_fraction"]), 3
                ),
            },
            "preview_jpeg_base64": preview_b64,
            "model_version": self.model_version,
            "confidence_is_calibrated": False,
            "possible_collapse_flag": False,
            "disclaimer": DISCLAIMER,
        }


def predict_image_bytes(data: bytes, filename: str) -> dict:
    """Decode an uploaded image and return its prediction."""

    array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ModelReliabilityError(
            "Could not decode the uploaded image."
        )

    predictor = get_predictor()

    return predictor.predict_bgr(bgr, filename)


_PREDICTOR: Predictor | None = None


def get_predictor() -> Predictor:
    """Return the cached prediction model."""

    global _PREDICTOR

    if _PREDICTOR is None:
        _PREDICTOR = Predictor()

    return _PREDICTOR
