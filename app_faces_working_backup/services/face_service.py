"""Lightweight local face detection and person grouping.

This module intentionally avoids dlib/face_recognition so the project remains
portable on Python 3.13 and macOS. OpenCV's bundled Haar cascade detects faces;
a compact DCT descriptor is used to group visually similar faces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.database import get_conn


class FaceServiceError(RuntimeError):
    """Raised when a photo cannot be processed for faces."""


@dataclass(frozen=True)
class DetectedFace:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    embedding: list[float]


_CASCADE: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _CASCADE
    if _CASCADE is None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise FaceServiceError(f"Unable to load OpenCV face cascade: {cascade_path}")
        _CASCADE = cascade
    return _CASCADE


def _face_embedding(gray_face: np.ndarray) -> list[float]:
    """Create a normalized appearance descriptor using low-frequency DCT values."""
    normalized = cv2.resize(gray_face, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = cv2.equalizeHist(normalized)
    dct = cv2.dct(normalized.astype(np.float32) / 255.0)
    vector = dct[:16, :16].reshape(-1)
    # Drop the DC brightness term and L2-normalize.
    vector = vector[1:]
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(float).tolist()


def detect_faces(path: Path) -> list[DetectedFace]:
    image = cv2.imread(str(path))
    if image is None:
        raise FaceServiceError(f"Unable to read image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    min_side = max(28, min(image.shape[:2]) // 12)
    boxes = _get_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_side, min_side),
    )

    faces: list[DetectedFace] = []
    for x, y, width, height in boxes:
        # Add a small margin so the descriptor includes the full face outline.
        margin_x = int(width * 0.12)
        margin_y = int(height * 0.12)
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(gray.shape[1], x + width + margin_x)
        y1 = min(gray.shape[0], y + height + margin_y)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        faces.append(
            DetectedFace(
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                confidence=1.0,
                embedding=_face_embedding(crop),
            )
        )
    return faces


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    if a.size != b.size or a.size == 0:
        return -1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def _person_centroids(conn) -> dict[int, list[float]]:
    grouped: dict[int, list[np.ndarray]] = {}
    rows = conn.execute(
        "SELECT person_id, embedding FROM faces WHERE person_id IS NOT NULL"
    ).fetchall()
    for row in rows:
        try:
            vector = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        grouped.setdefault(int(row["person_id"]), []).append(vector)

    centroids: dict[int, list[float]] = {}
    for person_id, vectors in grouped.items():
        same_size = [item for item in vectors if item.size == vectors[0].size]
        centroid = np.mean(np.stack(same_size), axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid /= norm
        centroids[person_id] = centroid.astype(float).tolist()
    return centroids


def _create_person(conn) -> int:
    cursor = conn.execute("INSERT INTO people(label) VALUES (NULL)")
    person_id = int(cursor.lastrowid)
    conn.execute("UPDATE people SET label = ? WHERE id = ?", (f"Person {person_id}", person_id))
    return person_id


def _choose_person(conn, embedding: list[float], threshold: float = 0.82) -> int:
    centroids = _person_centroids(conn)
    best_person: int | None = None
    best_score = -1.0
    for person_id, centroid in centroids.items():
        score = _cosine_similarity(embedding, centroid)
        if score > best_score:
            best_person = person_id
            best_score = score
    if best_person is not None and best_score >= threshold:
        return best_person
    return _create_person(conn)


def replace_photo_faces(conn, photo_id: int, path: Path, threshold: float = 0.82) -> int:
    """Re-detect faces for one photo and persist person assignments."""
    conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
    detections = detect_faces(path)
    for face in detections:
        person_id = _choose_person(conn, face.embedding, threshold=threshold)
        conn.execute(
            """
            INSERT INTO faces
            (photo_id, person_id, bbox_x, bbox_y, bbox_width, bbox_height,
             confidence, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                photo_id,
                person_id,
                face.x,
                face.y,
                face.width,
                face.height,
                face.confidence,
                json.dumps(face.embedding),
            ),
        )
    conn.execute("UPDATE photos SET face_count = ? WHERE id = ?", (len(detections), photo_id))
    return len(detections)


def rebuild_people(threshold: float = 0.82) -> dict:
    """Reprocess all indexed photos and rebuild person groups from scratch."""
    processed = 0
    faces_found = 0
    errors: list[dict] = []
    with get_conn() as conn:
        conn.execute("DELETE FROM faces")
        conn.execute("DELETE FROM people")
        conn.execute("UPDATE photos SET face_count = 0")
        photos = conn.execute("SELECT id, path FROM photos ORDER BY id").fetchall()
        for photo in photos:
            try:
                count = replace_photo_faces(
                    conn,
                    int(photo["id"]),
                    Path(photo["path"]),
                    threshold=threshold,
                )
                processed += 1
                faces_found += count
            except Exception as exc:
                errors.append({"path": photo["path"], "error": str(exc)})
    return {
        "processed": processed,
        "faces_found": faces_found,
        "people_created": people_count(),
        "errors": errors,
    }


def people_count() -> int:
    with get_conn() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0])
