"""Load labels and create leakage-safe stratified splits."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CLASS_NAMES,
    DATA_DIR,
    IMAGE_DIR,
    LABELS_CSV,
    PROCESSED_DIR,
    SEED,
    TEST_SPLIT,
    VALIDATION_SPLIT,
    ensure_runtime_dirs,
)
from src.data_validation import resolve_image_path  # noqa: E402

SAMPLE_ID_PATTERN = re.compile(r"^(Ch\d+_\d+)", re.IGNORECASE)


def sample_id_from_name(image_name: str) -> str:
    match = SAMPLE_ID_PATTERN.match(str(image_name))
    return match.group(1) if match else str(image_name)


def load_labelled_frame() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    df["full_path"] = [
        str(resolve_image_path(str(path), str(name)))
        for path, name in zip(df["image_path"], df["image_name"])
    ]
    df = df[df["full_path"].apply(lambda p: Path(p).is_file())].copy()
    df = df[df["grade"].isin(CLASS_NAMES)].copy()
    conflicts = df.groupby("full_path")["grade"].nunique()
    conflict_paths = conflicts[conflicts > 1].index
    df = df[~df["full_path"].isin(conflict_paths)].copy()
    df = df.drop_duplicates(subset="full_path").copy()
    df["sample_id"] = df["image_name"].map(sample_id_from_name)
    df["label"] = df["grade"].map({name: i for i, name in enumerate(CLASS_NAMES)}).astype(int)
    return df.reset_index(drop=True)


def _split_by_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = (
        df.groupby("sample_id")
        .agg(grade=("grade", "first"), label=("label", "first"))
        .reset_index()
    )
    test_size = TEST_SPLIT
    val_size = VALIDATION_SPLIT
    temp_size = test_size + val_size

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=temp_size,
        random_state=SEED,
        stratify=samples["label"],
    )
    relative_test = test_size / temp_size
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=relative_test,
        random_state=SEED,
        stratify=temp_samples["label"],
    )

    train_df = df[df["sample_id"].isin(train_samples["sample_id"])].copy()
    val_df = df[df["sample_id"].isin(val_samples["sample_id"])].copy()
    test_df = df[df["sample_id"].isin(test_samples["sample_id"])].copy()
    return train_df, val_df, test_df


def create_or_load_splits(reuse_existing: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prefer the previously published MobileNet splits when they exist so
    new experiments remain comparable and do not reshuffle the test set.
    """
    ensure_runtime_dirs()
    existing = (
        DATA_DIR / "mobilenet_train_split.csv",
        DATA_DIR / "mobilenet_validation_split.csv",
        DATA_DIR / "mobilenet_test_split.csv",
    )
    if reuse_existing and all(path.exists() for path in existing):
        frames = [pd.read_csv(path) for path in existing]
        for frame in frames:
            frame["full_path"] = [
                str(IMAGE_DIR / Path(str(name)).name)
                for name in frame["image_name"]
            ]
            missing = [p for p in frame["full_path"] if not Path(p).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} split images were not found, e.g. {missing[0]}"
                )
            frame["label"] = frame["grade"].map(
                {name: i for i, name in enumerate(CLASS_NAMES)}
            ).astype(int)
        train_df, val_df, test_df = frames
    else:
        df = load_labelled_frame()
        train_df, val_df, test_df = _split_by_sample(df)

    overlap = (
        set(train_df["image_name"])
        & set(val_df["image_name"])
        & set(test_df["image_name"])
    )
    train_val = set(train_df["image_name"]) & set(val_df["image_name"])
    train_test = set(train_df["image_name"]) & set(test_df["image_name"])
    val_test = set(val_df["image_name"]) & set(test_df["image_name"])
    if train_val or train_test or val_test or overlap:
        raise RuntimeError(
            "Data leakage detected across splits: "
            f"train/val={len(train_val)} train/test={len(train_test)} val/test={len(val_test)}"
        )

    train_df.assign(split="train").to_csv(PROCESSED_DIR / "train_split.csv", index=False)
    val_df.assign(split="validation").to_csv(PROCESSED_DIR / "val_split.csv", index=False)
    test_df.assign(split="test").to_csv(PROCESSED_DIR / "test_split.csv", index=False)
    return train_df, val_df, test_df
