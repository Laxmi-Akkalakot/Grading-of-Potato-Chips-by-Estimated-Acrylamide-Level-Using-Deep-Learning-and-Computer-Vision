"""Reliability checks for training, evaluation, and prediction."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class ModelReliabilityError(RuntimeError):
    """Raised when a model or prediction cannot be trusted."""


def detect_collapse(
    predicted_labels: Sequence[int] | np.ndarray,
    n_classes: int,
    min_classes: int = 2,
) -> dict:
    """Detect single-class prediction collapse on an evaluation set."""
    preds = np.asarray(predicted_labels).reshape(-1)
    if preds.size == 0:
        raise ModelReliabilityError("Empty prediction array.")
    unique = np.unique(preds)
    collapsed = unique.size < min_classes
    return {
        "collapsed": bool(collapsed),
        "unique_class_indices": unique.tolist(),
        "n_classes_expected": int(n_classes),
        "dominant_class_index": int(np.bincount(preds.astype(int)).argmax())
        if preds.size
        else None,
        "message": (
            "MODEL COLLAPSE: predictions used fewer than "
            f"{min_classes} classes."
            if collapsed
            else "Predictions cover multiple classes."
        ),
    }


def validate_scores(scores: np.ndarray) -> np.ndarray:
    """Reject NaN/Inf/empty softmax (or logit) arrays."""
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0:
        raise ModelReliabilityError("Empty prediction scores.")
    if not np.isfinite(array).all():
        raise ModelReliabilityError("Prediction scores contain NaN or Inf.")
    return array


def validate_class_mapping(class_names: Iterable[str], expected: Sequence[str]) -> list[str]:
    names = list(class_names)
    if not names:
        raise ModelReliabilityError("Class name list is empty.")
    if len(names) != len(set(names)):
        raise ModelReliabilityError("Duplicate class names in mapping.")
    missing = [name for name in expected if name not in names]
    extra = [name for name in names if name not in expected]
    if missing or extra:
        raise ModelReliabilityError(
            "Class mapping mismatch. "
            f"missing={missing} extra={extra} loaded={names}"
        )
    return names


def backup_if_exists(path: Path) -> Path | None:
    """Copy an existing model file before overwriting it."""
    path = Path(path)
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}_backup_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
