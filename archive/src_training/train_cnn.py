"""Train a small custom CNN for four-class acrylamide-grade classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CLASS_NAMES, IMAGE_SIZE, MODELS_DIR, NEW_CNN_PATH, RESULTS_DIR  # noqa: E402
from src.dataset import create_or_load_splits  # noqa: E402
from src.tf_datasets import make_tf_dataset, training_augmentation  # noqa: E402
from src.train_utils import (  # noqa: E402
    check_val_collapse,
    class_weights_if_needed,
    default_epochs,
    make_callbacks,
    save_model_bundle,
    set_seed,
)


def build_custom_cnn(n_classes: int = 4) -> tf.keras.Model:
    aug = training_augmentation()
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = aug(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def preprocess(image: tf.Tensor) -> tf.Tensor:
    return tf.cast(image, tf.float32) / 255.0


def main(epochs: int | None = None) -> Path:
    set_seed()
    train_df, val_df, test_df = create_or_load_splits()
    print("Split sizes:", len(train_df), len(val_df), len(test_df))
    print("Class index order:", CLASS_NAMES)
    print("Test set is held out and is not used for model selection.")

    train_ds = make_tf_dataset(train_df, preprocess, training=True)
    val_ds = make_tf_dataset(val_df, preprocess, training=False)

    model = build_custom_cnn(len(CLASS_NAMES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history_csv = RESULTS_DIR / "metrics" / "custom_cnn_history.csv"
    callbacks = make_callbacks(NEW_CNN_PATH, history_csv)
    weights = class_weights_if_needed(train_df)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=default_epochs(epochs),
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )
    save_model_bundle(model, NEW_CNN_PATH, "cnn_scale", extra={"architecture": "custom_cnn"})
    check_val_collapse(model, val_ds, "custom_cnn")
    return NEW_CNN_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.epochs)
