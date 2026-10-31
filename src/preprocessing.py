"""Shared image loading and preprocessing for train / val / test / predict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from config import IMAGE_SIZE

PreprocessName = Literal["cnn_scale", "mobilenet_v2", "efficientnet"]


def load_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def validate_image_array(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Empty image array.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image dimensions: {image.shape}")
    if min(image.shape[:2]) < 16:
        raise ValueError(f"Image is too small: {image.shape}")
    return image


def bgr_to_rgb_resized(bgr: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    validate_image_array(bgr)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def enhance_for_preview(bgr: np.ndarray) -> np.ndarray:
    """Mild contrast-limited enhancement for visual previews only."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    merged = cv2.merge((lightness, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_named_preprocess(rgb_uint8: np.ndarray, name: PreprocessName) -> np.ndarray:
    """
    Convert a resized RGB uint8 image into the float tensor a model expects.

    Training and prediction must use the same name stored in preprocess_config.json.
    """
    image = rgb_uint8.astype(np.float32)
    if name == "cnn_scale":
        return image / 255.0
    if name == "mobilenet_v2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        return preprocess_input(image)
    if name == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input

        return preprocess_input(image)
    raise ValueError(f"Unknown preprocess name: {name}")


def image_to_model_batch(
    bgr: np.ndarray,
    preprocess_name: PreprocessName,
    size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    rgb = bgr_to_rgb_resized(bgr, size)
    tensor = apply_named_preprocess(rgb, preprocess_name)
    return np.expand_dims(tensor, axis=0)


def save_preprocess_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_preprocess_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
