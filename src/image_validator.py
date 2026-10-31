
import cv2
import numpy as np
import tensorflow as tf

from config import MODELS_DIR
from src.safeguards import ModelReliabilityError

MODEL_PATH = (
    MODELS_DIR
    / "image_validation"
    / "potato_chip_validator.keras"
)

_model = None


def get_validator():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise ModelReliabilityError(
                "Potato-chip image validator model is missing."
            )

        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        # Warm up the model once.
        dummy = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        _model(dummy, training=False)

    return _model


def validate_potato_chip(bgr_image):
    if bgr_image is None or bgr_image.size == 0:
        raise ModelReliabilityError(
            "Invalid image. Please upload a valid photograph."
        )

    model = get_validator()

    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(np.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    image = np.expand_dims(image, axis=0)

    prediction = model(image, training=False)
    probability = float(prediction.numpy()[0][0])

    return {
        "is_potato_chip": probability >= 0.5,
        "potato_chip_probability": round(probability, 4),
        "validation_status": (
            "passed" if probability >= 0.5 else "rejected"
        )
    }
