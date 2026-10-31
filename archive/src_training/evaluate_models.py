"""Evaluate saved classifiers on the untouched test split and write reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CLASS_NAMES,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    NEW_CNN_PATH,
    NEW_EFFICIENTNET_PATH,
    NEW_MOBILENET_PATH,
    REPORTS_DIR,
    ensure_runtime_dirs,
)
from src.dataset import create_or_load_splits  # noqa: E402
from src.safeguards import detect_collapse, validate_scores  # noqa: E402


def _sklearn_metrics(y_true, y_pred, y_prob, class_names: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    roc = None
    try:
        roc = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        roc = None
    collapse = detect_collapse(y_pred, n_classes=len(class_names))
    per_class_recall = {
        class_names[i]: float(report[class_names[i]]["recall"])
        for i in range(len(class_names))
        if class_names[i] in report
    }
    weak = [name for name, rec in per_class_recall.items() if rec < 0.4]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": roc,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
        "collapse": collapse,
        "weak_recall_classes": weak,
        "per_class_recall": per_class_recall,
    }


def _plot_history(csv_path: Path, title: str, out_stub: str) -> None:
    if not csv_path.exists():
        return
    try:
        import matplotlib.pyplot as plt

        hist = pd.read_csv(csv_path)
        if "accuracy" in hist.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(hist["accuracy"], label="train accuracy")
            if "val_accuracy" in hist.columns:
                ax.plot(hist["val_accuracy"], label="val accuracy")
            ax.set_title(f"{title} accuracy")
            ax.set_xlabel("Epoch")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / f"{out_stub}_accuracy.png", dpi=140)
            plt.close(fig)
        if "loss" in hist.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(hist["loss"], label="train loss")
            if "val_loss" in hist.columns:
                ax.plot(hist["val_loss"], label="val loss")
            ax.set_title(f"{title} loss")
            ax.set_xlabel("Epoch")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / f"{out_stub}_loss.png", dpi=140)
            plt.close(fig)
    except Exception as exc:
        print("History plot skipped:", exc)


def _plot_confusion(matrix: list, class_names: list[str], title: str, out_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            np.array(matrix),
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / out_name, dpi=140)
        plt.close(fig)
    except Exception as exc:
        print("Confusion plot skipped:", exc)


def evaluate_keras_model(model_path: Path, preprocess_name: str, class_names: list[str], split_name: str, frame) -> dict | None:
    if not model_path.exists():
        print("Missing model, skip:", model_path)
        return None

    import tensorflow as tf

    from src.tf_datasets import make_tf_dataset

    if preprocess_name == "cnn_scale":
        def preprocess(image):
            return tf.cast(image, tf.float32) / 255.0
    elif preprocess_name == "mobilenet_v2":
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif preprocess_name == "efficientnet":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(preprocess_name)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        print(f"Failed to load {model_path}: {exc}")
        return None

    ds = make_tf_dataset(frame, preprocess, training=False)
    probs = model.predict(ds, verbose=0)
    validate_scores(probs)
    y_pred = np.argmax(probs, axis=1)
    y_true = frame["label"].to_numpy()
    metrics = _sklearn_metrics(y_true, y_pred, probs, class_names)
    metrics["model_path"] = str(model_path)
    metrics["split"] = split_name
    metrics["n"] = int(len(frame))
    return metrics


def existing_published_metrics() -> list[dict]:
    """Load previously computed test metrics; never fabricate numbers."""
    path = ROOT / "evaluation" / "performance_metrics_comparison.csv"
    acc_path = ROOT / "evaluation" / "accuracy_comparison.csv"
    if not path.exists():
        return []
    perf = pd.read_csv(path)
    acc = pd.read_csv(acc_path) if acc_path.exists() else None
    rows = []
    for _, row in perf.iterrows():
        item = {
            "model": str(row["Model"]),
            "source": "evaluation/performance_metrics_comparison.csv",
            "f1_macro": float(row["Macro F1-score (%)"]) / 100.0,
            "precision_macro": float(row["Macro Precision (%)"]) / 100.0,
            "recall_macro": float(row["Macro Recall (%)"]) / 100.0,
            "f1_weighted": float(row["Weighted F1-score (%)"]) / 100.0,
            "split": "test_published",
        }
        if acc is not None:
            match = acc[acc["Model"] == row["Model"]]
            if not match.empty:
                item["accuracy"] = float(match.iloc[0]["Accuracy (%)"]) / 100.0
        item["collapse"] = str(row["Model"]).strip() == "Original CNN"
        rows.append(item)
    return rows


def write_dashboard_bundle(new_results: list[dict]) -> None:
    published = existing_published_metrics()
    comparison_rows = []
    for item in published:
        comparison_rows.append(
            {
                "model": item["model"],
                "accuracy": item.get("accuracy"),
                "f1_macro": item.get("f1_macro"),
                "f1_weighted": item.get("f1_weighted"),
                "source": item["source"],
                "collapse": bool(item.get("collapse", False)),
            }
        )
    for item in new_results:
        comparison_rows.append(
            {
                "model": item["model"],
                "accuracy": item["test"]["accuracy"] if "test" in item else item.get("accuracy"),
                "f1_macro": item["test"]["f1_macro"] if "test" in item else item.get("f1_macro"),
                "f1_weighted": item["test"]["f1_weighted"] if "test" in item else item.get("f1_weighted"),
                "source": item.get("model_path", "new_evaluation"),
                "collapse": item.get("test", {}).get("collapse", {}).get("collapsed", False),
            }
        )

    best = None
    for item in new_results:
        val_f1 = item.get("validation", {}).get("f1_macro")
        if val_f1 is None:
            continue
        if best is None or val_f1 > best["validation"]["f1_macro"]:
            best = item
    if best is None and published:
        published_sorted = sorted(
            published, key=lambda x: x.get("f1_macro") or 0, reverse=True
        )
        top = published_sorted[0]
        best_payload = {
            "model": top["model"],
            "selection_basis": "highest published test macro F1 (no new validation run)",
            "test_accuracy": top.get("accuracy"),
            "test_macro_f1": top.get("f1_macro"),
            "model_path": str(MODELS_DIR / "acrylamide_grade_v3.keras"),
            "class_names_file": str(MODELS_DIR / "grade_classes_v3.json"),
            "preprocess": "mobilenet_v2",
        }
    elif best is not None:
        best_payload = {
            "model": best["model"],
            "selection_basis": "highest validation macro F1; test reported separately",
            "validation_macro_f1": best["validation"]["f1_macro"],
            "test_accuracy": best["test"]["accuracy"],
            "test_macro_f1": best["test"]["f1_macro"],
            "model_path": best["model_path"],
            "preprocess": best["preprocess"],
            "collapse_on_test": best["test"]["collapse"]["collapsed"],
        }
    else:
        best_payload = {
            "model": None,
            "error": "No evaluated model available.",
        }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    (METRICS_DIR / "best_model.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    (METRICS_DIR / "dashboard.json").write_text(
        json.dumps(
            {
                "best_model": best_payload,
                "comparison": comparison_rows,
                "disclaimer": (
                    "Metrics are computed from saved evaluation files or a held-out test set. "
                    "They are not laboratory measurement accuracy."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt

        names = [row["model"] for row in comparison_rows if row.get("accuracy") is not None]
        accs = [100 * row["accuracy"] for row in comparison_rows if row.get("accuracy") is not None]
        if names:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(names, accs, color="#2f6f4e")
            ax.set_xlabel("Test accuracy (%)")
            ax.set_title("Model comparison (from saved evaluations)")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=140)
            plt.close(fig)
    except Exception as exc:
        print("Comparison chart skipped:", exc)


def main() -> None:
    ensure_runtime_dirs()
    train_df, val_df, test_df = create_or_load_splits()
    candidates = [
        ("Custom CNN", NEW_CNN_PATH, "cnn_scale", "custom_cnn"),
        ("MobileNetV2", NEW_MOBILENET_PATH, "mobilenet_v2", "mobilenetv2"),
        ("EfficientNetB0", NEW_EFFICIENTNET_PATH, "efficientnet", "efficientnetb0"),
    ]
    new_results = []
    for name, path, preprocess_name, stub in candidates:
        val_metrics = evaluate_keras_model(path, preprocess_name, CLASS_NAMES, "validation", val_df)
        test_metrics = evaluate_keras_model(path, preprocess_name, CLASS_NAMES, "test", test_df)
        if test_metrics is None:
            continue
        if test_metrics["collapse"]["collapsed"]:
            print(f"WARNING: {name} collapsed on the test set.")
        if test_metrics["weak_recall_classes"]:
            print(f"WARNING: {name} weak recall:", test_metrics["weak_recall_classes"])

        report_df = pd.DataFrame(test_metrics["classification_report"]).T
        report_df.to_csv(METRICS_DIR / f"{stub}_classification_report.csv")
        pd.DataFrame(
            test_metrics["confusion_matrix"],
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_csv(METRICS_DIR / f"{stub}_confusion_matrix.csv")
        _plot_confusion(
            test_metrics["confusion_matrix"],
            CLASS_NAMES,
            f"{name} test confusion matrix",
            f"{stub}_confusion_matrix.png",
        )
        _plot_history(METRICS_DIR / f"{stub}_history.csv", name, stub)
        new_results.append(
            {
                "model": name,
                "model_path": str(path),
                "preprocess": preprocess_name,
                "validation": val_metrics,
                "test": test_metrics,
            }
        )

    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    (REPORTS_DIR / "new_model_evaluation.json").write_text(
        json.dumps(new_results, indent=2, cls=_Encoder), encoding="utf-8"
    )
    write_dashboard_bundle(new_results)
    print("Wrote metrics to", METRICS_DIR)


if __name__ == "__main__":
    main()
