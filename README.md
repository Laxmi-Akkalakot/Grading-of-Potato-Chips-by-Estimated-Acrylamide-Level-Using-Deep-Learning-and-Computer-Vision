# PotatoGuard AI – Smart Acrylamide Grading System

Grading of Potato Chips by Estimated Acrylamide Level Using Deep Learning and Computer Vision.

This repository is an academic major project. It estimates a **project-defined acrylamide grade** from potato-chip images. Image predictions are **not** laboratory measurements.

## Problem statement

Acrylamide can form in potato products during high-temperature frying. Laboratory methods remain the reference for concentration. This project studies whether computer vision and deep learning can estimate a four-band **visual grade** that was assigned from laboratory values already stored in the dataset.

## Objectives

- Validate an existing labelled image dataset (do not invent labels).
- Extract colour and brightness features after chip segmentation.
- Train and compare a custom CNN, MobileNetV2, and EfficientNetB0.
- Optionally estimate concentration (µg/kg) from colour features.
- Serve a Flask web application for demonstration and history logging.

## Existing dataset

The original dataset is preserved in `data/`.

| Item | Location |
| --- | --- |
| Labels | `data/labels.csv` |
| Images | `data/images/` (310 JPG files) |
| Laboratory workbook | `data/Acrylamide_dataset_with_image_names.xlsx` |

Expected columns: `image_name`, `image_path`, `acrylamide_ug_per_kg`, `grade`, `excel_band`, `source_sheet`.

Project-defined grades (not official safety limits):

1. Low: below 500 µg/kg  
2. Medium: 500 to below 1000 µg/kg  
3. High: 1000 to below 2000 µg/kg  
4. Very High: 2000 µg/kg and above  

Approximate class counts: Low 126, Medium 99, High 56, Very High 29.

## Technology stack

Python 3.10+, TensorFlow/Keras, OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, Flask, SQLite, HTML/CSS/JavaScript, Bootstrap 5.

## System architecture

1. Dataset validation reads `labels.csv` and files in `data/images/`.
2. Preprocessing resizes images to 160×160 RGB and applies a model-specific normalisation stored next to each `.keras` file.
3. Segmentation and feature extraction support analysis and regression.
4. Classifiers output four-class softmax scores (uncalibrated).
5. Flask serves the dashboard and `/api/predict`. SQLite stores prediction history only (not the image files).

## Methodology

- Stratified 70 / 15 / 15 split. Existing MobileNet split CSVs are reused when present so the published test set is not reshuffled.
- Split membership is checked for image-name leakage.
- Augmentation (flip, small rotation, zoom, contrast) is training-only.
- Class weights are applied only when the train-set majority/minority ratio is at least 2.
- Early stopping, learning-rate reduction, and best-checkpoint saving are enabled.
- Model selection for newly trained networks uses **validation** macro F1. Test metrics are reported afterwards.
- Single-class prediction collapse is detected automatically.

## Training instructions (Windows PowerShell)

```powershell
cd "C:\Users\siddu\OneDrive\Documents\GitHub\Grading-of-Potato-Chips-by-Estimated-Acrylamide-Level-Using-Deep-Learning-and-Computer-Vision"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src\data_validation.py
python src\feature_extraction.py
python src\train_cnn.py --epochs 20
python src\train_transfer_learning.py --backbone mobilenetv2 --epochs 20
python src\train_transfer_learning.py --backbone efficientnetb0 --epochs 20
python src\train_regression.py
python src\evaluate_models.py
```

Existing trained files such as `models/acrylamide_grade_v3.keras` are **not** overwritten by these scripts. New weights are written to `models/custom_cnn.keras`, `models/mobilenetv2.keras`, and `models/efficientnet.keras`. If a target file already exists, a timestamped backup is created first.

## Evaluation methodology

Metrics: accuracy, precision, recall, F1 (macro and weighted), confusion matrix, classification report, and ROC-AUC (one-vs-rest) when scores allow it.

Regression metrics: MAE, RMSE, R² on laboratory µg/kg targets from the dataset (never fabricated).

Published comparison (held-out test, 47 images), from `evaluation/`:

| Model | Test accuracy | Macro F1 | Notes |
| --- | --- | --- | --- |
| Original CNN | 40.43% | 14.39% | Collapse: predicted only Low on the test set |
| MobileNetV2 V2 | 74.47% | 66.77% | |
| Fine-tuned MobileNetV2 V3 | 78.72% | 71.13% | Selected published model |

Very High has only four test images, so that class estimate is unstable. Medium recall for V3 was 66.7%; Very High recall was 50%. High overall accuracy must not be read as success for every class.

## How to run the web application

```powershell
python app.py
```

Then open http://127.0.0.1:5000

The app does **not** retrain on startup. If the classifier file is missing or corrupt, the UI shows an error and does not silently load a different model.

## REST API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health` | App and classifier status |
| GET | `/api/dataset` | Counts from `labels.csv` |
| GET | `/api/metrics` | Metrics from evaluation files |
| POST | `/api/predict` | Multipart form field `image` |
| GET | `/api/history` | Recent SQLite rows |
| DELETE | `/api/history/<id>` | Delete one history row |

Example:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/predict -F "image=@data\images\Ch3_001_Low_267.jpg"
```

## Limitations

- 310 images, with a small Very High group.
- Image estimates cannot replace laboratory testing.
- Softmax scores are not calibrated probabilities.
- Colour/brightness are visual descriptors, not acrylamide concentration.
- The original custom CNN collapsed to a single class; that failure is reported, not hidden.

## Future enhancements

More original images (especially Very High), independent test collections, probability calibration, and additional backbones.

## Licence

See `LICENSE`.
