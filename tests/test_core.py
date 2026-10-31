"""Lightweight checks that do not require a trained neural network."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CLASS_NAMES, grade_from_acrylamide
from src.dataset import create_or_load_splits, load_labelled_frame
from src.safeguards import detect_collapse


class GradeMappingTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(grade_from_acrylamide(0), "Low")
        self.assertEqual(grade_from_acrylamide(499.9), "Low")
        self.assertEqual(grade_from_acrylamide(500), "Medium")
        self.assertEqual(grade_from_acrylamide(999.9), "Medium")
        self.assertEqual(grade_from_acrylamide(1000), "High")
        self.assertEqual(grade_from_acrylamide(1999.9), "High")
        self.assertEqual(grade_from_acrylamide(2000), "Very High")


class CollapseTests(unittest.TestCase):
    def test_detects_single_class(self):
        report = detect_collapse([0, 0, 0, 0], n_classes=4)
        self.assertTrue(report["collapsed"])

    def test_multi_class_ok(self):
        report = detect_collapse([0, 1, 2, 3], n_classes=4)
        self.assertFalse(report["collapsed"])


class DatasetTests(unittest.TestCase):
    def test_labels_exist_and_classes_match(self):
        df = load_labelled_frame()
        self.assertGreaterEqual(len(df), 300)
        self.assertTrue(set(df["grade"].unique()).issubset(set(CLASS_NAMES)))

    def test_splits_have_no_leakage(self):
        train_df, val_df, test_df = create_or_load_splits()
        train, val, test = set(train_df["image_name"]), set(val_df["image_name"]), set(test_df["image_name"])
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)


if __name__ == "__main__":
    unittest.main()
