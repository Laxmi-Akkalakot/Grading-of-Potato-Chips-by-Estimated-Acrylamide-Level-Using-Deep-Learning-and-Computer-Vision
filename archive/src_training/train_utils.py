"""Shared training helpers: class weights, callbacks, collapse checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import CLASS_NAMES, EPOCHS, MODELS_DIR, SEED
from src.safeguards import backup_if_exists, detect_collapse, write_json


def set_seed(seed: int = SEED) -> None:
    tf.keras.utils.set_random_seed(seed)


def class_weights_if_needed(train_df: pd.DataFrame, imbalance_ratio: float = 2.0) -> dict[int, float] | None:
    """Use balanced class weights only when the majority/minority ratio is large."""
    counts = train_df["label"].value_counts()
    if counts.min() == 0:
        return None
    ratio = float(counts.max()) / float(counts.min())
    print("Train class counts:\n", counts.sort_index())
    print(f"Majority/minority ratio: {ratio:.2f}")
    if ratio < imbalance_ratio:
        print("Class weights not applied (imbalance below threshold).")
        return None
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_df["label"].to_numpy(),
    )
    mapping = {int(i): float(w) for i, w in enumerate(weights)}
    print("Applying class weights:", mapping)
    return mapping


def make_callbacks(model_path: Path, history_csv: Path) -> list:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(history_csv)),
    ]


def save_model_bundle(
    model: tf.keras.Model,
    model_path: Path,
    preprocess_name: str,
    extra: dict | None = None,
) -> None:
    backup = backup_if_exists(model_path)
    if backup:
        print("Backed up existing model to", backup)
    model.save(model_path)
    payload = {
        "class_names": CLASS_NAMES,
        "image_size": [160, 160],
        "preprocess": preprocess_name,
        "color_mode": "rgb",
        "label_index_order": CLASS_NAMES,
    }
    if extra:
        payload.update(extra)
    write_json(model_path.with_name(model_path.stem + "_preprocess.json"), payload)
    write_json(MODELS_DIR / "grade_classes.json", {"class_names": CLASS_NAMES})
    print("Saved", model_path)


def check_val_collapse(model: tf.keras.Model, val_ds, name: str) -> dict:
    probs = model.predict(val_ds, verbose=0)
    preds = np.argmax(probs, axis=1)
    report = detect_collapse(preds, n_classes=len(CLASS_NAMES))
    print(f"[{name}] validation collapse check:", report["message"])
    return report


def default_epochs(cli_epochs: int | None) -> int:
    return int(cli_epochs) if cli_epochs else EPOCHS
