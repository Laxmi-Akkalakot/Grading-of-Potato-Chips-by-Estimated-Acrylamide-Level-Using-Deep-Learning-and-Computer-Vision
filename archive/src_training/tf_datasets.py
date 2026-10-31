"""TensorFlow datasets with training-only augmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from config import BATCH_SIZE, IMAGE_SIZE, SEED


def _decode_resize(path: tf.Tensor) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def make_tf_dataset(
    frame: pd.DataFrame,
    preprocess_fn,
    training: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    paths = frame["full_path"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int32)

    def load(path, label):
        image = _decode_resize(path)
        image = preprocess_fn(image)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def training_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.08),
        ],
        name="train_augmentation",
    )
