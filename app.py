
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageChops

# ==========================================
# 1. PROJECT SETTINGS
# ==========================================
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "acrylamide_grade.keras"
CLASSES_PATH = ROOT / "models" / "grade_classes.json"
LABELS_PATH = ROOT / "data" / "labels.csv"
IMAGES_PATH = ROOT / "data" / "images"

IMAGE_SIZE = (160, 160)

GRADE_LIMITS = [
    (500, "Low"),
    (1000, "Medium"),
    (2000, "High"),
    (float("inf"), "Very High"),
]

st.set_page_config(
    page_title="Potato Chip Acrylamide Analysis",
    page_icon="🥔",
    layout="wide",
)

# ==========================================
# 2. LOAD MODEL, CLASSES AND LABELS
# ==========================================
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        classes = json.load(file)

    # Support either a list or a dictionary in the JSON file.
    if isinstance(classes, dict):
        try:
            classes = [
                name for name, index in
                sorted(classes.items(), key=lambda item: int(item[1]))
            ]
        except (ValueError, TypeError):
            classes = list(classes.keys())

    return model, classes


@st.cache_data
def load_labels():
    if not LABELS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(LABELS_PATH)


def grade_from_measurement(value):
    if pd.isna(value):
        return "Unknown"

    for upper_limit, grade in GRADE_LIMITS:
        if float(value) < upper_limit:
            return grade

    return "Very High"


# ==========================================
# 3. IMAGE COLOUR ANALYSIS
# ==========================================
def analyse_colour(image):
    """
    Calculate approximate image-colour statistics.
    These are image features, not chemical measurements.
    Background pixels can affect the results.
    """
    rgb = np.asarray(
        image.convert("RGB").resize((300, 300)),
        dtype=np.float32,
    )

    # Approximate perceived brightness on a 0-255 scale.
    brightness = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )

    mean_r = float(np.mean(rgb[:, :, 0]))
    mean_g = float(np.mean(rgb[:, :, 1]))
    mean_b = float(np.mean(rgb[:, :, 2]))
    mean_brightness = float(np.mean(brightness))

    # Approximate brown-like pixels using RGB conditions.
    # This is not a calibrated colour measurement.
    brown_mask = (
        (rgb[:, :, 0] > rgb[:, :, 1])
        & (rgb[:, :, 1] > rgb[:, :, 2])
        & (rgb[:, :, 0] < 190)
        & (rgb[:, :, 1] < 150)
        & (rgb[:, :, 2] < 120)
    )

    brown_percent = float(np.mean(brown_mask) * 100)

    return {
        "brightness": mean_brightness,
        "red": mean_r,
        "green": mean_g,
        "blue": mean_b,
        "brown_percent": brown_percent,
    }


# ==========================================
# 4. VERIFY WHETHER IMAGE IS IN THE DATASET
# ==========================================
def find_exact_dataset_match(uploaded_image, uploaded_name, labels):
    """
    Show a laboratory reference only when the uploaded image
    exactly matches the original image file at the same name.
    """
    if labels.empty or not IMAGES_PATH.exists():
        return None

    if "image_name" not in labels.columns:
        return None

    rows = labels[
        labels["image_name"].astype(str).str.casefold()
        == uploaded_name.casefold()
    ]

    for _, row in rows.iterrows():
        image_path = IMAGES_PATH / str(row["image_name"])

        if not image_path.is_file():
            continue

        try:
            with Image.open(image_path) as reference:
                reference = reference.convert("RGB")
                candidate = uploaded_image.convert("RGB")

                if reference.size != candidate.size:
                    continue

                difference = ImageChops.difference(
                    reference, candidate
                )

                if difference.getbbox() is None:
                    return row

        except Exception:
            continue

    return None


# ==========================================
# 5. WEBSITE DESIGN
# ==========================================
st.title("🥔 Potato Chip Acrylamide Analysis")
st.subheader("Deep Learning and Computer Vision")

st.write(
    "Upload a potato-chip image to explore its visual colour "
    "features and obtain an experimental acrylamide-grade "
    "prediction from the trained CNN."
)

st.info(
    "Important: chip colour alone cannot establish the actual "
    "acrylamide concentration. A laboratory measurement is "
    "required to know the actual value."
)

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        "Model files are missing. Check the models folder and "
        "train the classifier before running the application."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a potato-chip image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

# ==========================================
# 6. PREDICTION AND ANALYSIS
# ==========================================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width=420,
        )

        if st.button("Analyse Chip", type="primary"):
            with st.spinner("Analysing the image..."):
                model, classes = load_model_and_classes()

                resized = image.resize(IMAGE_SIZE)

                image_array = np.asarray(
                    resized, dtype=np.float32
                ) / 255.0

                image_array = np.expand_dims(image_array, axis=0)

                probabilities = model.predict(
                    image_array, verbose=0
                )[0]

                if len(probabilities) != len(classes):
                    st.error(
                        "The model output does not match the "
                        "saved class names. Check grade_classes.json."
                    )
                    st.stop()

                best_index = int(np.argmax(probabilities))
                predicted_grade = str(classes[best_index])
                confidence = float(probabilities[best_index] * 100)

                colour = analyse_colour(image)
                labels = load_labels()

                reference = find_exact_dataset_match(
                    image, uploaded_file.name, labels
                )

            # ----------------------------------
            # CNN RESULT
            # ----------------------------------
            st.divider()
            st.header("1. AI prediction")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted grade", predicted_grade)

            with col2:
                st.metric(
                    "Model confidence score",
                    f"{confidence:.2f}%",
                )

            st.caption(
                "Confidence is the model's output score, not "
                "the probability that the result is scientifically "
                "correct."
            )

            st.subheader("Prediction scores by grade")

            for name, score in sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True,
            ):
                st.write(f"**{name}: {score * 100:.2f}%**")
                st.progress(float(score))

            # ----------------------------------
            # IMAGE COLOUR FEATURES
            # ----------------------------------
            st.divider()
            st.header("2. Image colour analysis")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Mean brightness",
                f"{colour['brightness']:.1f} / 255",
            )
            c2.metric(
                "Mean RGB red channel",
                f"{colour['red']:.1f}",
            )
            c3.metric(
                "Approx. brown-like pixels",
                f"{colour['brown_percent']:.1f}%",
            )

            with st.expander("View all image colour features"):
                st.write(
                    {
                        "Mean red channel": round(colour["red"], 2),
                        "Mean green channel": round(colour["green"], 2),
                        "Mean blue channel": round(colour["blue"], 2),
                        "Mean brightness": round(
                            colour["brightness"], 2
                        ),
                        "Approx. brown-like pixel percentage": round(
                            colour["brown_percent"], 2
                        ),
                    }
                )

            st.warning(
                "These are approximate whole-image colour statistics. "
                "The background, shadows and lighting can affect them. "
                "They are not acrylamide measurements or calibrated "
                "L*a*b* measurements."
            )

            # ----------------------------------
            # LABORATORY REFERENCE
            # ----------------------------------
            st.divider()
            st.header("3. Laboratory reference comparison")

            if reference is not None:
                actual_value = float(
                    reference["acrylamide_ug_per_kg"]
                )
                actual_grade = grade_from_measurement(actual_value)

                r1, r2 = st.columns(2)

                with r1:
                    st.metric(
                        "Measured acrylamide",
                        f"{actual_value:.2f} µg/kg",
                    )

                with r2:
                    st.metric(
                        "Grade from measured value",
                        actual_grade,
                    )

                if predicted_grade.casefold() == actual_grade.casefold():
                    st.success(
                        "The CNN prediction matches the "
                        "laboratory-derived grade for this known sample."
                    )
                else:
                    st.error(
                        "The CNN prediction differs from the "
                        "laboratory-derived grade for this known sample."
                    )

                st.caption(
                    "This reference is shown because the uploaded image "
                    "exactly matches a labelled image in your dataset. "
                    "It is not a new laboratory test."
                )

            else:
                st.info(
                    "No exact match was found in the labelled image "
                    "dataset. The actual acrylamide value is unknown "
                    "for this upload. Only the experimental CNN "
                    "prediction is available."
                )

            # ----------------------------------
            # SCIENTIFIC INTERPRETATION
            # ----------------------------------
            st.divider()
            st.header("4. How to interpret this result")

            st.write(
                "A chip's colour is one visual feature. The actual "
                "acrylamide concentration can also depend on the "
                "potato raw material and processing conditions. "
                "Therefore, similar-looking chips may have different "
                "measured concentrations."
            )

            st.warning(
                "This academic model is experimental. It does not "
                "measure acrylamide directly and cannot certify food "
                "safety. Laboratory testing is required to establish "
                "the actual concentration."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")

else:
    st.info("Upload an image to get started.")

st.divider()
st.caption(
    "Academic project | Deep Learning and Computer Vision"
)
