"""Train MobileNetV2 or EfficientNetB0 transfer-learning classifiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CLASS_NAMES,
    IMAGE_SIZE,
    NEW_EFFICIENTNET_PATH,
    NEW_MOBILENET_PATH,
    RESULTS_DIR,
)
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


def _backbone(name: str):
    if name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(*IMAGE_SIZE, 3),
        )
        preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
        out_path = NEW_MOBILENET_PATH
        preprocess_name = "mobilenet_v2"
    elif name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(*IMAGE_SIZE, 3),
        )
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
        out_path = NEW_EFFICIENTNET_PATH
        preprocess_name = "efficientnet"
    else:
        raise ValueError("backbone must be mobilenetv2 or efficientnetb0")
    return base, preprocess_fn, out_path, preprocess_name


def build_model(backbone_name: str) -> tuple[tf.keras.Model, Path, str, object]:
    base, preprocess_fn, out_path, preprocess_name = _backbone(backbone_name)
    base.trainable = False
    aug = training_augmentation()
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = aug(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{backbone_name}_classifier")
    return model, out_path, preprocess_name, preprocess_fn


def main(backbone: str = "mobilenetv2", epochs: int | None = None) -> Path:
    set_seed()
    train_df, val_df, test_df = create_or_load_splits()
    print("Split sizes:", len(train_df), len(val_df), len(test_df))
    print("Held-out test images:", len(test_df), "(not used for selection)")

    model, out_path, preprocess_name, preprocess_fn = build_model(backbone)
    train_ds = make_tf_dataset(train_df, preprocess_fn, training=True)
    val_ds = make_tf_dataset(val_df, preprocess_fn, training=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history_csv = RESULTS_DIR / "metrics" / f"{backbone}_history.csv"
    callbacks = make_callbacks(out_path, history_csv)
    weights = class_weights_if_needed(train_df)

    n_epochs = default_epochs(epochs)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Light fine-tune of the last convolutional block.
    base = next(
        layer for layer in model.layers if isinstance(layer, tf.keras.Model)
    )
    base.trainable = True
    freeze_until = max(0, len(base.layers) - 30)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(4, n_epochs // 3),
        class_weight=weights,
        callbacks=make_callbacks(out_path, RESULTS_DIR / "metrics" / f"{backbone}_finetune_history.csv"),
        verbose=1,
    )

    save_model_bundle(
        model,
        out_path,
        preprocess_name,
        extra={"architecture": backbone, "fine_tuned": True},
    )
    check_val_collapse(model, val_ds, backbone)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.backbone, args.epochs)
