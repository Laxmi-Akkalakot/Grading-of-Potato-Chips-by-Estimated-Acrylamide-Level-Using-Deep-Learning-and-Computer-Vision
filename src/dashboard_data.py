
"""Read dashboard numbers from dataset and evaluation files."""

from __future__ import annotations

import json

import pandas as pd

from config import (
    CLASS_NAMES,
    FIGURES_DIR,
    LABELS_CSV,
    METRICS_DIR,
    ROOT_DIR,
)


def dataset_overview() -> dict:
    if not LABELS_CSV.exists():
        return {
            "total_images": 0,
            "class_counts": {name: 0 for name in CLASS_NAMES},
            "acrylamide_min": None,
            "acrylamide_max": None,
            "acrylamide_median": None,
            "source_sheets": {},
        }

    df = pd.read_csv(LABELS_CSV)
    counts = df["grade"].value_counts().reindex(
        CLASS_NAMES
    ).fillna(0).astype(int)

    acrylamide = pd.to_numeric(
        df["acrylamide_ug_per_kg"], errors="coerce"
    )

    return {
        "total_images": int(len(df)),
        "class_counts": counts.to_dict(),
        "acrylamide_min": float(acrylamide.min()) if acrylamide.notna().any() else None,
        "acrylamide_max": float(acrylamide.max()) if acrylamide.notna().any() else None,
        "acrylamide_median": float(acrylamide.median()) if acrylamide.notna().any() else None,
        "source_sheets": (
            df["source_sheet"].value_counts().to_dict()
            if "source_sheet" in df.columns else {}
        ),
    }


def find_evaluation_file(filename: str):
    """Find an evaluation file in the main or archived evaluation folders."""
    candidates = [
        ROOT_DIR / "evaluation" / filename,
        ROOT_DIR / "evaluation" / "archive_visuals" / filename,
        ROOT_DIR / "evaluation" / "archive" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    # Search nested evaluation folders as a final fallback.
    matches = list((ROOT_DIR / "evaluation").rglob(filename))
    return matches[0] if matches else None


def performance_bundle() -> dict:
    dashboard_path = METRICS_DIR / "dashboard.json"

    if dashboard_path.exists():
        try:
            return json.loads(dashboard_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    published = find_evaluation_file(
        "performance_metrics_comparison.csv"
    )
    accuracy = find_evaluation_file(
        "accuracy_comparison.csv"
    )

    comparison = []

    if published and accuracy:
        try:
            perf = pd.read_csv(published)
            acc = pd.read_csv(accuracy)
            merged = perf.merge(acc, on="Model", how="left")

            for _, row in merged.iterrows():
                if pd.isna(row.get("Accuracy (%)")):
                    continue

                comparison.append({
                    "model": str(row["Model"]),
                    "accuracy": float(row["Accuracy (%)"]) / 100.0,
                    "f1_macro": float(row["Macro F1-score (%)"]) / 100.0,
                    "f1_weighted": float(row["Weighted F1-score (%)"]) / 100.0,
                    "source": str(published),
                    "collapse": str(row["Model"]) == "Original CNN",
                })

        except (KeyError, ValueError, OSError) as exc:
            print("Evaluation metrics could not be read:", exc)

    best = None

    if comparison:
        comparison_sorted = sorted(
            comparison,
            key=lambda item: item["f1_macro"],
            reverse=True,
        )

        best = {
            "model": comparison_sorted[0]["model"],
            "test_accuracy": comparison_sorted[0]["accuracy"],
            "test_macro_f1": comparison_sorted[0]["f1_macro"],
            "selection_basis": "Highest published macro F1 from evaluation files.",
        }

    return {
        "best_model": best or {
            "model": None,
            "test_accuracy": None,
            "test_macro_f1": None,
            "error": "Evaluation metrics are currently unavailable.",
        },
        "comparison": comparison,
        "disclaimer": "Values loaded from evaluation CSV files.",
    }


def figure_available(name: str) -> bool:
    return (
        (FIGURES_DIR / name).exists()
        or (ROOT_DIR / "evaluation" / name).exists()
    )
