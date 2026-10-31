"""Potato-chip segmentation helpers used by feature extraction and the web app."""

from __future__ import annotations

import cv2
import numpy as np


def segment_chip(bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask (uint8, 0/255) for the likely chip region.

    The method uses HSV colour thresholding plus the largest contour.
    It is a visual foreground estimate, not a chemical measurement.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty image passed to segment_chip.")

    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Potato-chip colours typically occupy yellow/brown hues.
    lower = np.array([5, 25, 25], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
        return clean

    # Fallback: use the central region so feature extraction still runs.
    h, w = mask.shape
    clean[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = 255
    return clean


def apply_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep chip pixels; set background to black."""
    return cv2.bitwise_and(bgr, bgr, mask=mask)
