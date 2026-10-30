
import os
import numpy as np
import tensorflow as tf
import cv2
import joblib

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from io import BytesIO

# --------------------------------------------------
# 1. Flask application configuration
# --------------------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSIFICATION_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "acrylamide_grade_v3.keras"
)

REGRESSION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "regression",
    "random_forest_acrylamide.pkl"
)

IMAGE_SIZE = (160, 160)

CLASS_NAMES = ["High", "Low", "Medium", "Very High"]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# --------------------------------------------------
# 2. Load classification model
# --------------------------------------------------

if not os.path.exists(CLASSIFICATION_MODEL_PATH):
    raise FileNotFoundError(CLASSIFICATION_MODEL_PATH)

print("Loading V3 classification model...")

classification_model = tf.keras.models.load_model(
    CLASSIFICATION_MODEL_PATH,
    compile=False
)

print("Classification model loaded successfully!")


# --------------------------------------------------
# 3. Load regression model
# --------------------------------------------------

if not os.path.exists(REGRESSION_MODEL_PATH):
    raise FileNotFoundError(REGRESSION_MODEL_PATH)

print("Loading Random Forest regression model...")

regression_model = joblib.load(REGRESSION_MODEL_PATH)

print("Regression model loaded successfully!")


# --------------------------------------------------
# 4. Check uploaded image extension
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# 5. Preprocess image for MobileNetV2
# --------------------------------------------------

def preprocess_classification_image(image_bytes):

    img = Image.open(BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)

    image_array = np.array(img, dtype=np.float32)

    image_array = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )
    )

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# 6. Extract image features for Random Forest
# --------------------------------------------------

def extract_regression_features(image_bytes):

    img = Image.open(BytesIO(image_bytes))
    img = img.convert("RGB")

    # Convert PIL RGB image to OpenCV BGR
    img_array = np.array(img)

    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (160, 160))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    # BGR channel statistics
    for channel in cv2.split(img):
        features.extend([
            np.mean(channel),
            np.std(channel)
        ])

    # HSV channel statistics
    for channel in cv2.split(hsv):
        features.extend([
            np.mean(channel),
            np.std(channel)
        ])

    # Brightness statistics
    features.extend([
        np.mean(gray),
        np.std(gray)
    ])

    return np.array(features).reshape(1, -1)


# --------------------------------------------------
# 7. Predict acrylamide grade
# --------------------------------------------------

def predict_grade(image_bytes):

    image_array = preprocess_classification_image(
        image_bytes
    )

    predictions = classification_model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_grade = CLASS_NAMES[predicted_index]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    return predicted_grade, confidence


# --------------------------------------------------
# 8. Predict numerical acrylamide value
# --------------------------------------------------

def predict_acrylamide_value(image_bytes):

    features = extract_regression_features(image_bytes)

    prediction = regression_model.predict(features)[0]

    prediction = max(0.0, float(prediction))

    return prediction


# --------------------------------------------------
# 9. Home page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# 10. Prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "Please select an image."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Only JPG, JPEG and PNG images are allowed."
        }), 400

    try:

        image_bytes = file.read()

        # Validate uploaded image
        img = Image.open(BytesIO(image_bytes))
        img.verify()

        # Predict grade using MobileNetV2
        predicted_grade, confidence = predict_grade(
            image_bytes
        )

        # Predict numerical value using Random Forest
        acrylamide_value = predict_acrylamide_value(
            image_bytes
        )

        # Return both predictions
        return jsonify({
            "success": True,
            "grade": predicted_grade,
            "confidence": round(confidence, 2),
            "acrylamide_value": round(acrylamide_value, 2),
            "unit": "µg/kg",
            "message": "Prediction completed successfully.",
            "disclaimer": (
                "Experimental image-based estimate. "
                "Not a direct chemical measurement."
            )
        })

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": (
                "Invalid or corrupted image. "
                "Please upload a valid image."
            )
        }), 400

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": "Prediction failed. Please try again."
        }), 500


# --------------------------------------------------
# 11. Run Flask application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
