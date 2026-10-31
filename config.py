"""
Central configuration for PotatoGuard AI.

All training, evaluation, and web-app pipelines should import these
constants so preprocessing and class labels stay consistent.
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PREVIEWS_DIR = PROCESSED_DIR / "previews"

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
METRICS_DIR = RESULTS_DIR / "metrics"

DATABASE_DIR = ROOT_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "app.db"

UPLOAD_DIR = ROOT_DIR / "uploads"

# Semantic grade order used by new training scripts.
CLASS_NAMES = ["Low", "Medium", "High", "Very High"]

# Project-defined category thresholds (µg/kg). These are not official
# regulatory safety limits.
GRADE_THRESHOLDS = {
    "Low": (0.0, 500.0),
    "Medium": (500.0, 1000.0),
    "High": (1000.0, 2000.0),
    "Very High": (2000.0, float("inf")),
}

IMAGE_SIZE = (160, 160)
IMAGE_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# WEBP enabled (changed from {"png", "jpg", "jpeg"})
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_MB = 8

# Preferred production classifier. Existing v3 is reused until a newer
# evaluated model is registered in results/metrics/best_model.json.
LEGACY_CLASSIFIER_PATH = MODELS_DIR / "acrylamide_grade_v3.keras"
LEGACY_CLASSES_PATH = MODELS_DIR / "grade_classes_v3.json"
LEGACY_REGRESSION_PATH = MODELS_DIR / "regression" / "random_forest_acrylamide.pkl"

NEW_CNN_PATH = MODELS_DIR / "custom_cnn.keras"
NEW_MOBILENET_PATH = MODELS_DIR / "mobilenetv2.keras"
NEW_EFFICIENTNET_PATH = MODELS_DIR / "efficientnet.keras"
GRADE_CLASSES_PATH = MODELS_DIR / "grade_classes.json"

PREPROCESS_CONFIG_NAME = "preprocess_config.json"

DISCLAIMER = (
    "Predictions are research-based estimates from image appearance. "
    "They cannot replace laboratory acrylamide testing."
)

APP_NAME = "PotatoGuard AI – Smart Acrylamide Grading System"
APP_SHORT_NAME = "PotatoGuard AI"
MODEL_VERSION_DEFAULT = "unknown"


def ensure_runtime_dirs() -> None:
    """Create output folders used by training, evaluation, and the app."""
    for path in (
        PROCESSED_DIR,
        PREVIEWS_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        METRICS_DIR,
        DATABASE_DIR,
        UPLOAD_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def grade_from_acrylamide(value: float) -> str:
    """Map a laboratory acrylamide value (µg/kg) to a project grade."""
    if value < 500:
        return "Low"
    if value < 1000:
        return "Medium"
    if value < 2000:
        return "High"
    return "Very High"