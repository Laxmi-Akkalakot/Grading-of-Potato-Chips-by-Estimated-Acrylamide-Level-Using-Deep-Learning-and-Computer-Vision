"""PotatoGuard AI Flask application."""

from __future__ import annotations

import os
from io import BytesIO

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from PIL import Image, UnidentifiedImageError

from config import (
    ALLOWED_EXTENSIONS,
    APP_NAME,
    DISCLAIMER,
    FIGURES_DIR,
    MAX_UPLOAD_MB,
    RESULTS_DIR,
    ROOT_DIR,
)
from src.dashboard_data import dataset_overview, performance_bundle
from src.db import delete_prediction, init_db, insert_prediction, list_predictions
from src.safeguards import ModelReliabilityError

# Set to False to turn off the debug printout in the terminal.
DEBUG_PREDICTIONS = True

app = Flask(__name__)
app.secret_key = os.environ.get("POTATOGUARD_SECRET", "potatoguard-dev-key")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_predictor = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_formats_text() -> str:
    names = sorted(ext.upper() for ext in ALLOWED_EXTENSIONS)
    return ", ".join(names)


def predictor_or_error():
    """Return (predictor, None) or (None, error). Only successes are cached."""
    global _predictor
    if _predictor is not None:
        return _predictor, None
    try:
        from src.predict import get_predictor

        _predictor = get_predictor()
        return _predictor, None
    except Exception as exc:
        print("Model load error:", exc)
        return None, str(exc)


def safe_call(func, default, *args, **kwargs):
    """Run a data helper; on failure print the error and return a default."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        print(f"{func.__name__} failed:", exc)
        return default


def debug_prediction(filename: str, image_bytes: bytes, result: dict) -> None:
    """Print prediction details to the terminal. Never raises."""
    if not DEBUG_PREDICTIONS:
        return
    try:
        print("\n========== PREDICTION DEBUG ==========")
        print("Filename:", filename)
        print("Image bytes:", len(image_bytes))
        print("Estimated value:", result.get("estimated_acrylamide_ug_per_kg"))
        print("Predicted grade:", result.get("grade"))
        print("Class confidences:", result.get("class_confidences"))
        print("Model version:", result.get("model_version"))
        print("Extracted features:", result.get("features"))
        print("======================================\n")
    except Exception as exc:
        print("Debug print failed:", exc)


@app.context_processor
def inject_globals():
    return {
        "app_name": APP_NAME,
        "disclaimer": DISCLAIMER,
    }


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": f"Image too large. Maximum size is {MAX_UPLOAD_MB} MB."}), 413


@app.route("/")
def index():
    overview = safe_call(dataset_overview, {})
    performance = safe_call(performance_bundle, {})
    recent = safe_call(list_predictions, [], limit=5)
    return render_template(
        "index.html",
        dataset=overview,
        overview=overview,
        performance=performance,
        recent=recent,
        predictor_error=predictor_or_error()[1],
    )


@app.route("/upload")
def upload():
    return render_template("upload.html", predictor_error=predictor_or_error()[1])


@app.route("/camera")
def camera_page():
    return render_template("camera.html", predictor_error=predictor_or_error()[1])


@app.route("/results")
def results_page():
    payload = session.get("last_prediction")
    if not payload:
        flash("Upload an image to see prediction results.")
        return redirect(url_for("upload"))
    return render_template("results.html", result=payload)


@app.route("/analysis")
def analysis():
    payload = session.get("last_prediction")
    return render_template("analysis.html", result=payload)


@app.route("/dataset")
def dataset_page():
    overview = safe_call(dataset_overview, {})
    return render_template("dataset.html", dataset=overview, overview=overview)


@app.route("/performance")
def performance_page():
    performance = safe_call(performance_bundle, {})
    return render_template("performance.html", performance=performance)


@app.route("/history")
def history_page():
    return render_template("history.html", records=safe_call(list_predictions, []))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/health")
def api_health():
    _predictor_obj, error = predictor_or_error()
    return jsonify(
        {
            "ok": error is None,
            "model_error": error,
            "app": APP_NAME,
        }
    )


@app.route("/api/dataset")
def api_dataset():
    return jsonify(safe_call(dataset_overview, {}))


@app.route("/api/metrics")
def api_metrics():
    return jsonify(safe_call(performance_bundle, {}))


@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify(safe_call(list_predictions, []))


@app.route("/api/history/<int:item_id>", methods=["DELETE"])
def api_history_delete(item_id: int):
    if not delete_prediction(item_id):
        return jsonify({"error": "Prediction not found."}), 404
    return jsonify({"ok": True, "deleted": item_id})


@app.route("/history/<int:item_id>/delete", methods=["POST"])
def history_delete(item_id: int):
    delete_prediction(item_id)
    flash("Prediction deleted.")
    return redirect(url_for("history_page"))


@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def api_predict():
    predictor, error = predictor_or_error()
    if error or predictor is None:
        return jsonify({"error": error or "Classifier is unavailable."}), 503

    if "image" not in request.files:
        return jsonify({"error": "Please select an image."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    filename = file.filename
    # Camera captures may arrive without a file extension.
    if "." not in filename:
        filename = f"{filename}.jpg"

    if not allowed_file(filename):
        return jsonify(
            {"error": f"Only {allowed_formats_text()} images are allowed."}
        ), 400

    image_bytes = file.read()
    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "Invalid or corrupted image."}), 400

    try:
        from src.predict import predict_image_bytes

        result = predict_image_bytes(image_bytes, filename)
    except ModelReliabilityError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print("Prediction error:", exc)
        return jsonify({"error": "Prediction failed. Please try again."}), 500

    debug_prediction(filename, image_bytes, result)

    try:
        insert_prediction(
            filename=result["filename"],
            grade=result["grade"],
            confidences=result["class_confidences"],
            model_version=result["model_version"],
            estimated_ug_per_kg=result.get("estimated_acrylamide_ug_per_kg"),
        )
    except Exception as exc:
        print("Could not save prediction to history:", exc)

    session["last_prediction"] = {
        key: value
        for key, value in result.items()
        if key != "preview_jpeg_base64"
    }
    return jsonify({"success": True, **result})


@app.route("/results/figures/<path:filename>")
def result_figure(filename: str):
    directory = FIGURES_DIR
    target = directory / filename
    if not target.exists():
        abort(404)
    return send_from_directory(directory, filename)


@app.route("/evaluation/<path:filename>")
def evaluation_file(filename: str):
    directory = ROOT_DIR / "evaluation"
    if not (directory / filename).exists():
        abort(404)
    return send_from_directory(directory, filename)


def create_app() -> Flask:
    init_db()
    return app


if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False)