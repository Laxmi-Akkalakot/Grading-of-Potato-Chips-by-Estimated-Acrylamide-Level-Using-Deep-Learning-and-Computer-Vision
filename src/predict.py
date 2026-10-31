"""Load the acrylamide regression model and predict a single potato-chip image."""

from __future__ import annotations

import base64
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

    IMPORTANT:
    Keep this feature order identical to the original training code.
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

    return np.asarray([features], dtype=np.float64)


def _rounded(features: dict, key: str, digits: int = 2):
    """Return a rounded feature value, or None if it is missing."""
    value = features.get(key)
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


class Predictor:
    """Predict estimated acrylamide concentration from a potato-chip image."""

    def __init__(self) -> None:
        self.regression_bundle = None
        self.model_version = "Unknown"
        self._load()

    def _load(self) -> None:
        """Load the available regression model."""

        new_model_path = (
            MODELS_DIR / "regression" / "random_forest_features.joblib"
        )

        legacy_model_path = LEGACY_REGRESSION_PATH

        if new_model_path.exists():
            model_path = new_model_path
            default_legacy = False

        elif legacy_model_path.exists():
            model_path = legacy_model_path
            default_legacy = True

        else:
            raise ModelReliabilityError(
                "No regression model found. Expected either "
                f"{new_model_path} or {legacy_model_path}"
            )

        try:
            loaded = joblib.load(model_path)

            if isinstance(loaded, dict) and "model" in loaded:
                self.regression_bundle = loaded

                if "legacy" not in self.regression_bundle:
                    self.regression_bundle["legacy"] = default_legacy

            else:
                self.regression_bundle = {
                    "model": loaded,
                    "legacy": default_legacy,
                }

            self.model_version = model_path.name

            model = self.regression_bundle["model"]

            print("\n========== MODEL INFORMATION ==========")
            print("Model:", self.model_version)
            print("Legacy model:", self.regression_bundle["legacy"])
            print(
                "Expected input features:",
                getattr(model, "n_features_in_", "Not available")
            )
            print(
                "Feature columns:",
                self.regression_bundle.get("feature_columns", "Not available")
            )
            print("=======================================\n")

        except Exception as exc:
            raise ModelReliabilityError(
                f"Could not load regression model: {exc}"
            ) from exc

    def predict_bgr(
        self,
        bgr: np.ndarray,
        filename: str = "upload.jpg",
    ) -> dict:
        """Predict concentration and derive the project grade."""

        # 1. Validate input image.
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

        # 2. Extract segmented image features and preview.
        features, mask, preview = extract_features_from_bgr(bgr)

        bundle = self.regression_bundle
        model = bundle["model"]

        # 3. Select the exact input representation used during training.
        columns = bundle.get("feature_columns")

        if columns:
            missing = [
                column for column in columns
                if column not in features
            ]

            if missing:
                raise ModelReliabilityError(
                    f"Required model features are missing: {missing}"
                )

            vector = np.asarray(
                [[features[column] for column in columns]],
                dtype=np.float64,
            )

            input_mode = "Saved feature-column model"

            input_feature_names = list(columns)

        elif bundle.get("legacy", False):
            # This model was trained using the entire unsegmented image.
            vector = extract_legacy_rf_features(bgr)

            input_mode = "Legacy full-image Random Forest"

            input_feature_names = [
                "mean_blue", "std_blue",
                "mean_green", "std_green",
                "mean_red", "std_red",
                "mean_hue", "std_hue",
                "mean_saturation", "std_saturation",
                "mean_value", "std_value",
                "mean_gray", "std_gray",
            ]

        else:
            vector = regression_vector_from_features(features)

            input_mode = "Segmented feature model"

            input_feature_names = [
                "mean_blue", "std_blue",
                "mean_green", "std_green",
                "mean_red", "std_red",
                "mean_hue", "std_hue",
                "mean_saturation", "std_saturation",
                "mean_value", "std_value",
                "brightness", "brightness_std",
            ]

        # 4. Validate model input dimensions.
        expected_features = getattr(model, "n_features_in_", None)

        if (
            expected_features is not None
            and vector.shape[1] != expected_features
        ):
            raise ModelReliabilityError(
                "Model feature mismatch: model expects "
                f"{expected_features} features, but received "
                f"{vector.shape[1]}."
            )

        if not np.all(np.isfinite(vector)):
            raise ModelReliabilityError(
                "Invalid numerical values found in extracted features."
            )

        # 5. Predict numerical acrylamide estimate.
        try:
            raw_prediction = model.predict(vector)[0]

            estimated_value = float(raw_prediction)

            if not np.isfinite(estimated_value):
                raise ValueError("Model returned an invalid prediction.")

            estimated_value = max(0.0, estimated_value)

        except Exception as exc:
            raise ModelReliabilityError(
                f"Acrylamide prediction failed: {exc}"
            ) from exc

        # 6. Assign the project-defined grade.
        grade = grade_from_acrylamide(estimated_value)

        # 7. Encode processing preview.
        ok, preview_buffer = cv2.imencode(".jpg", preview)

        preview_b64 = None

        if ok:
            preview_b64 = base64.b64encode(
                preview_buffer.tobytes()
            ).decode("ascii")

        # 8. Print actual model input for debugging.
        model_input = {
            name: round(float(value), 4)
            for name, value in zip(
                input_feature_names,
                vector[0]
            )
        }

        print("\n========== PREDICTION DEBUG ==========")
        print("Filename:", filename)
        print("Image dimensions:", width, "x", height)
        print("Model:", self.model_version)
        print("Input mode:", input_mode)
        print("Input feature count:", vector.shape[1])
        print("Actual model input:", model_input)
        print("Estimated value:", round(estimated_value, 2))
        print("Predicted grade:", grade)
        print("=====================================\n")

        # 9. Return result to Flask.
        return {
            "filename": filename,

            "grade": grade,

            "confidence_percent": None,

            "class_confidences": {},

            "estimated_acrylamide_ug_per_kg": round(
                estimated_value, 2
            ),

            # Features shown on the dashboard.
            "features": {
                "mean_red": _rounded(features, "mean_red"),
                "mean_green": _rounded(features, "mean_green"),
                "mean_blue": _rounded(features, "mean_blue"),
                "mean_hue": _rounded(features, "mean_hue"),
                "mean_saturation": _rounded(features, "mean_saturation"),
                "mean_value": _rounded(features, "mean_value"),
                "brightness": _rounded(features, "brightness"),
                "saturation": _rounded(features, "saturation"),
                "foreground_fraction": _rounded(
                    features, "foreground_fraction", 3
                ),
            },

            # Actual features used by the regression model.
            "model_input_mode": input_mode,
            "model_input_features": model_input,

            "preview_jpeg_base64": preview_b64,

            "model_version": self.model_version,

            "confidence_is_calibrated": False,

            "possible_collapse_flag": False,

            "disclaimer": DISCLAIMER,
        }


def predict_image_bytes(data: bytes, filename: str) -> dict:
    """Validate potato-chip image before acrylamide prediction."""

    if not data:
        raise ModelReliabilityError("The uploaded image is empty.")

    array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ModelReliabilityError(
            "Could not decode the uploaded image."
        )

    # STEP 1: Validate whether the image contains potato chips.
    validation = None
    try:
        from src.image_validator import validate_potato_chip
    except ImportError:
        print(
            "\nWARNING: src/image_validator.py not found. "
            "Skipping potato-chip validation.\n"
        )
    else:
        validation = validate_potato_chip(bgr)

        print("\n========== IMAGE VALIDATION ==========")
        print("Filename:", filename)
        print(
            "Potato-chip probability:",
            validation.get("potato_chip_probability")
        )
        print("Validation:", validation.get("validation_status"))
        print("======================================\n")

        # STEP 2: Reject unrelated images.
        if not validation.get("is_potato_chip", False):
            raise ModelReliabilityError(
                "Image validation failed. No potato chips were detected. "
                "Please upload a clear potato-chip photograph."
            )

    # STEP 3: Run the acrylamide regression model.
    predictor = get_predictor()
    result = predictor.predict_bgr(bgr, filename)

    # STEP 4: Include image validation details in the result.
    if validation is not None:
        result["image_validation"] = validation

    return result


_PREDICTOR: Predictor | None = None


def get_predictor() -> Predictor:
    """Return the cached prediction model."""

    global _PREDICTOR

    if _PREDICTOR is None:
        _PREDICTOR = Predictor()

    return _PREDICTOR