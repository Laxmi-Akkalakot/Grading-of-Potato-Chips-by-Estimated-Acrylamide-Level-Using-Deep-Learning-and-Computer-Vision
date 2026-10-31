"""Validate labels.csv against files in data/images/. Never rewrites the dataset."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CLASS_NAMES,
    FIGURES_DIR,
    IMAGE_DIR,
    LABELS_CSV,
    REPORTS_DIR,
    ensure_runtime_dirs,
    grade_from_acrylamide,
)


SAMPLE_ID_PATTERN = re.compile(r"^(Ch\d+_\d+)", re.IGNORECASE)


def resolve_image_path(raw_path: str, image_name: str) -> Path:
    candidate = Path(str(raw_path).replace("\\", "/"))
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.is_file():
        return candidate
    return IMAGE_DIR / image_name


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset() -> dict:
    ensure_runtime_dirs()

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_CSV}")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing image directory: {IMAGE_DIR}")

    df = pd.read_csv(LABELS_CSV)
    expected_columns = {
        "image_name",
        "image_path",
        "acrylamide_ug_per_kg",
        "grade",
        "excel_band",
        "source_sheet",
    }
    missing_columns = sorted(expected_columns - set(df.columns))

    issues = {
        "missing_columns": missing_columns,
        "missing_images": [],
        "incorrect_image_paths": [],
        "corrupted_images": [],
        "invalid_acrylamide_values": [],
        "incorrect_grade_assignments": [],
        "duplicate_image_names": [],
        "duplicate_sample_ids": [],
        "duplicate_file_hashes": [],
    }

    image_files = {
        path.name: path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    labelled_names = set(df["image_name"].astype(str)) if "image_name" in df.columns else set()
    unlabelled_images = sorted(set(image_files) - labelled_names)

    hashes: dict[str, list[str]] = {}
    sample_ids: list[str] = []

    if not missing_columns:
        df["acrylamide_ug_per_kg"] = pd.to_numeric(
            df["acrylamide_ug_per_kg"], errors="coerce"
        )

        name_counts = df["image_name"].astype(str).value_counts()
        issues["duplicate_image_names"] = name_counts[name_counts > 1].index.tolist()

        for _, row in df.iterrows():
            name = str(row["image_name"])
            raw_path = str(row["image_path"])
            resolved = resolve_image_path(raw_path, name)
            listed = ROOT / Path(raw_path.replace("\\", "/"))

            match = SAMPLE_ID_PATTERN.match(name)
            sample_ids.append(match.group(1) if match else name)

            value = row["acrylamide_ug_per_kg"]
            if pd.isna(value) or value < 0:
                issues["invalid_acrylamide_values"].append(name)
                expected_grade = None
            else:
                expected_grade = grade_from_acrylamide(float(value))
                if str(row["grade"]) != expected_grade:
                    issues["incorrect_grade_assignments"].append(
                        {
                            "image_name": name,
                            "acrylamide_ug_per_kg": float(value),
                            "csv_grade": str(row["grade"]),
                            "expected_grade": expected_grade,
                        }
                    )

            if not resolved.is_file():
                issues["missing_images"].append(name)
                continue

            if listed.resolve() != resolved.resolve():
                issues["incorrect_image_paths"].append(
                    {
                        "image_name": name,
                        "csv_path": raw_path,
                        "resolved_path": str(resolved),
                    }
                )

            image = cv2.imread(str(resolved))
            if image is None or image.size == 0:
                issues["corrupted_images"].append(name)
                continue

            height, width = image.shape[:2]
            if height < 16 or width < 16:
                issues["corrupted_images"].append(name)

            digest = file_md5(resolved)
            hashes.setdefault(digest, []).append(name)

        id_counts = Counter(sample_ids)
        issues["duplicate_sample_ids"] = [
            sample_id for sample_id, count in id_counts.items() if count > 1
        ]
        issues["duplicate_file_hashes"] = [
            names for names in hashes.values() if len(names) > 1
        ]

    grade_counts = (
        df["grade"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict()
        if "grade" in df.columns
        else {}
    )

    report = {
        "labels_csv": str(LABELS_CSV),
        "image_dir": str(IMAGE_DIR),
        "n_label_rows": int(len(df)),
        "n_image_files": len(image_files),
        "unlabelled_images": unlabelled_images,
        "class_distribution": grade_counts,
        "source_sheets": (
            df["source_sheet"].value_counts().to_dict()
            if "source_sheet" in df.columns
            else {}
        ),
        "acrylamide_stats": {},
        "issues": issues,
        "ok": True,
    }

    if "acrylamide_ug_per_kg" in df.columns and df["acrylamide_ug_per_kg"].notna().any():
        series = pd.to_numeric(df["acrylamide_ug_per_kg"], errors="coerce")
        report["acrylamide_stats"] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
        }

    issue_counts = {
        key: len(value) for key, value in issues.items() if key != "missing_columns"
    }
    issue_counts["missing_columns"] = len(issues["missing_columns"])
    report["issue_counts"] = issue_counts
    report["ok"] = all(count == 0 for count in issue_counts.values()) and not unlabelled_images

    _write_reports(report, df if "grade" in df.columns else None)
    return report


def _write_reports(report: dict, df: pd.DataFrame | None) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS_DIR / "dataset_validation.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Dataset Validation Report",
        "",
        f"- Label rows: {report['n_label_rows']}",
        f"- Image files: {report['n_image_files']}",
        f"- Validation passed: {report['ok']}",
        "",
        "## Class distribution",
        "",
    ]
    for grade, count in report["class_distribution"].items():
        lines.append(f"- {grade}: {count}")

    lines.extend(["", "## Issue counts", ""])
    for key, count in report["issue_counts"].items():
        lines.append(f"- {key}: {count}")

    if report["unlabelled_images"]:
        lines.extend(["", "## Unlabelled image files", ""])
        lines.extend(f"- {name}" for name in report["unlabelled_images"])

    if report["issues"]["incorrect_grade_assignments"]:
        lines.extend(["", "## Grade mismatches (CSV grade vs value-based rule)", ""])
        for item in report["issues"]["incorrect_grade_assignments"][:50]:
            lines.append(
                f"- {item['image_name']}: value={item['acrylamide_ug_per_kg']:.2f}, "
                f"csv={item['csv_grade']}, expected={item['expected_grade']}"
            )

    (REPORTS_DIR / "dataset_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if df is not None and "grade" in df.columns:
        try:
            import matplotlib.pyplot as plt

            counts = df["grade"].value_counts().reindex(CLASS_NAMES).fillna(0)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(counts.index.astype(str), counts.values, color="#2f6f4e")
            ax.set_title("Dataset class distribution")
            ax.set_ylabel("Number of images")
            ax.set_xlabel("Acrylamide grade (project-defined)")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=140)
            plt.close(fig)
        except Exception as exc:  # plotting must not fail validation
            print("Could not save class-distribution figure:", exc)


def main() -> None:
    report = validate_dataset()
    print(json.dumps({
        "ok": report["ok"],
        "n_label_rows": report["n_label_rows"],
        "n_image_files": report["n_image_files"],
        "class_distribution": report["class_distribution"],
        "issue_counts": report["issue_counts"],
        "unlabelled_images": report["unlabelled_images"],
    }, indent=2))
    print("Wrote", REPORTS_DIR / "dataset_validation.md")


if __name__ == "__main__":
    main()
