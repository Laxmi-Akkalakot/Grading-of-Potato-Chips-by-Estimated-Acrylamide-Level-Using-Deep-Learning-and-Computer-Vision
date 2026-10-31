"""Extract RGB/HSV colour statistics from segmented potato-chip images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    FIGURES_DIR,
    IMAGE_DIR,
    LABELS_CSV,
    PREVIEWS_DIR,
    PROCESSED_DIR,
    ensure_runtime_dirs,
)
from src.data_validation import resolve_image_path  # noqa: E402
from src.preprocessing import enhance_for_preview, load_bgr  # noqa: E402
from src.segmentation import apply_mask, segment_chip  # noqa: E402


def colour_histogram(channel: np.ndarray, bins: int = 8) -> list[float]:
    hist, _ = np.histogram(channel, bins=bins, range=(0, 256), density=True)
    return hist.astype(np.float64).tolist()


def extract_features_from_bgr(bgr: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return features, chip mask, and a preview image with mask overlay."""
    working = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA)
    mask = segment_chip(working)
    masked = apply_mask(working, mask)
    pixels = working[mask > 0]
    if len(pixels) < 50:
        pixels = working.reshape(-1, 3)

    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv[mask > 0]
    if len(hsv_pixels) < 50:
        hsv_pixels = hsv.reshape(-1, 3)

    b, g, r = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray_pixels = gray[mask > 0] if (mask > 0).any() else gray.reshape(-1)

    features = {
        "mean_red": float(np.mean(r)),
        "mean_green": float(np.mean(g)),
        "mean_blue": float(np.mean(b)),
        "std_red": float(np.std(r)),
        "std_green": float(np.std(g)),
        "std_blue": float(np.std(b)),
        "mean_hue": float(np.mean(h)),
        "mean_saturation": float(np.mean(s)),
        "mean_value": float(np.mean(v)),
        "std_hue": float(np.std(h)),
        "std_saturation": float(np.std(s)),
        "std_value": float(np.std(v)),
        "brightness": float(np.mean(gray_pixels)),
        "brightness_std": float(np.std(gray_pixels)),
        "saturation": float(np.mean(s)),
        "foreground_fraction": float(np.mean(mask > 0)),
        "rgb_hist": colour_histogram(r) + colour_histogram(g) + colour_histogram(b),
        "hsv_hist": colour_histogram(h) + colour_histogram(s) + colour_histogram(v),
    }

    overlay = working.copy()
    colour_mask = np.zeros_like(working)
    colour_mask[:, :] = (40, 180, 80)
    overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(working, 0.75, colour_mask, 0.25, 0), overlay)
    preview = np.hstack([working, enhance_for_preview(working), masked, overlay])
    return features, mask, preview


def regression_vector_from_features(features: dict) -> np.ndarray:
    """Numeric vector compatible with the existing Random Forest feature layout."""
    return np.array(
        [
            features["mean_blue"],
            features["std_blue"],
            features["mean_green"],
            features["std_green"],
            features["mean_red"],
            features["std_red"],
            features["mean_hue"],
            features["std_hue"],
            features["mean_saturation"],
            features["std_saturation"],
            features["mean_value"],
            features["std_value"],
            features["brightness"],
            features["brightness_std"],
        ],
        dtype=np.float64,
    ).reshape(1, -1)


def extract_dataset_features(max_previews: int = 12) -> pd.DataFrame:
    ensure_runtime_dirs()
    df = pd.read_csv(LABELS_CSV)
    rows = []
    preview_count = 0

    for _, row in df.iterrows():
        name = str(row["image_name"])
        path = resolve_image_path(str(row["image_path"]), name)
        if not path.is_file():
            continue
        try:
            bgr = load_bgr(path)
            features, _, preview = extract_features_from_bgr(bgr)
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
            continue

        record = {
            "image_name": name,
            "image_path": str(path),
            "acrylamide_ug_per_kg": row.get("acrylamide_ug_per_kg"),
            "grade": row.get("grade"),
        }
        for key, value in features.items():
            if key in {"rgb_hist", "hsv_hist"}:
                continue
            record[key] = value
        for i, value in enumerate(features["rgb_hist"]):
            record[f"rgb_hist_{i}"] = value
        for i, value in enumerate(features["hsv_hist"]):
            record[f"hsv_hist_{i}"] = value
        rows.append(record)

        if preview_count < max_previews:
            cv2.imwrite(str(PREVIEWS_DIR / f"preview_{preview_count:02d}_{name}"), preview)
            preview_count += 1

    out = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "image_features.csv"
    out.to_csv(out_path, index=False)
    note = {
        "note": (
            "Colour and brightness features describe image appearance. "
            "They are not laboratory measurements of acrylamide."
        ),
        "n_rows": int(len(out)),
        "output": str(out_path),
    }
    (PROCESSED_DIR / "image_features_readme.json").write_text(
        json.dumps(note, indent=2), encoding="utf-8"
    )
    _plot_brightness(out)
    print(f"Saved {len(out)} feature rows to {out_path}")
    return out


def _plot_brightness(df: pd.DataFrame) -> None:
    if df.empty or "brightness" not in df.columns:
        return
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        grades = ["Low", "Medium", "High", "Very High"]
        data = [df.loc[df["grade"] == grade, "brightness"].dropna().values for grade in grades]
        ax.boxplot(data, labels=grades)
        ax.set_title("Image brightness by labelled grade")
        ax.set_ylabel("Mean brightness (0-255)")
        ax.set_xlabel("Grade (laboratory-derived labels)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "brightness_by_grade.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print("Could not save brightness figure:", exc)


def main() -> None:
    extract_dataset_features()


if __name__ == "__main__":
    main()
