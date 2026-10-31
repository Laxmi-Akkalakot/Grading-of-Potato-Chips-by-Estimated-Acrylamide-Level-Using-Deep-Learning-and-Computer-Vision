"""SQLite helpers for prediction history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from config import DATABASE_PATH, ensure_runtime_dirs


def get_connection() -> sqlite3.Connection:
    ensure_runtime_dirs()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT NOT NULL,
                predicted_grade TEXT NOT NULL,
                confidence_scores TEXT NOT NULL,
                estimated_ug_per_kg REAL,
                model_version TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def insert_prediction(
    filename: str,
    grade: str,
    confidences: dict,
    model_version: str,
    estimated_ug_per_kg: float | None,
) -> int:
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                image_filename, predicted_grade, confidence_scores,
                estimated_ug_per_kg, model_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                grade,
                json.dumps(confidences),
                estimated_ug_per_kg,
                model_version,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_predictions(limit: int = 100) -> list[dict]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["confidence_scores"] = json.loads(item["confidence_scores"])
        results.append(item)
    return results


def delete_prediction(prediction_id: int) -> bool:
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        conn.commit()
        return cursor.rowcount > 0
