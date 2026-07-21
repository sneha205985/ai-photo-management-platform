"""CLIP-based image understanding and semantic search."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"

CATEGORY_PROMPTS: dict[str, str] = {
    "document": "a photo of a document, form, certificate, or official paperwork",
    "prescription": "a photo of a medical prescription, medicine bottle, or pharmacy label",
    "receipt": "a photo of a store receipt, restaurant bill, or purchase invoice",
    "people": "a photo of a person, portrait, selfie, or group of people",
    "travel": "a photo of travel, vacation, landmarks, beaches, or scenic destinations",
    "pet": "a photo of a pet such as a dog, cat, puppy, or kitten",
    "other": "a photo of miscellaneous objects or an uncategorized scene",
}


class ClipServiceError(Exception):
    """Raised when the CLIP model is unavailable or inference fails."""


class ClipService:
    """Lazy-loaded singleton wrapper around openai/clip-vit-base-patch32."""

    _instance: ClipService | None = None
    _init_lock = Lock()

    def __new__(cls) -> ClipService:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._ready = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_ready", False):
            return
        self._model: Any = None
        self._processor: Any = None
        self._load_error: str | None = None
        self._model_lock = Lock()
        self._ready = True

    @property
    def available(self) -> bool:
        return self._model is not None and self._processor is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        if self._load_error is not None:
            raise ClipServiceError(self._load_error)

        with self._model_lock:
            if self._model is not None and self._processor is not None:
                return
            if self._load_error is not None:
                raise ClipServiceError(self._load_error)
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                logger.info("Loading CLIP model %s (first run may download weights)...", MODEL_ID)
                self._processor = CLIPProcessor.from_pretrained(MODEL_ID)
                self._model = CLIPModel.from_pretrained(MODEL_ID)
                self._model.eval()
                logger.info("CLIP model loaded successfully.")
            except Exception as exc:
                self._load_error = str(exc)
                logger.warning("Failed to load CLIP model: %s", exc)
                raise ClipServiceError(str(exc)) from exc

    def encode_image(self, path: Path) -> list[float]:
        self._ensure_loaded()
        import torch

        with Image.open(path) as image:
            rgb = image.convert("RGB")
        inputs = self._processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).tolist()

    def encode_text(self, text: str) -> list[float]:
        self._ensure_loaded()
        import torch

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).tolist()

    def classify_image(self, path: Path) -> tuple[str, str]:
        self._ensure_loaded()
        import torch

        with Image.open(path) as image:
            rgb = image.convert("RGB")

        labels = list(CATEGORY_PROMPTS.keys())
        prompts = [CATEGORY_PROMPTS[label] for label in labels]
        inputs = self._processor(text=prompts, images=rgb, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image.squeeze(0)
            best_idx = int(logits.argmax().item())

        category = labels[best_idx]
        description = f"Image classified as {category} using CLIP zero-shot classification."
        return category, description


def get_clip_service() -> ClipService:
    return ClipService()


def serialize_embedding(embedding: list[float]) -> str:
    return json.dumps(embedding)


def deserialize_embedding(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    return json.loads(raw)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def analyze_image(path: Path) -> tuple[str, str, str | None]:
    """Return category, description, and serialized embedding using CLIP."""
    clip = get_clip_service()
    category, description = clip.classify_image(path)
    embedding = clip.encode_image(path)
    return category, description, serialize_embedding(embedding)


def semantic_search(query: str, limit: int = 50) -> list[dict]:
    """Hybrid CLIP + OCR search ranked by a combined relevance score."""
    from app.core.database import get_conn

    clip = get_clip_service()
    query_embedding = clip.encode_text(query)
    query_terms = {term for term in query.lower().split() if len(term) >= 3}

    results: list[dict] = []
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM photos").fetchall()

    for row in rows:
        photo = dict(row)
        image_embedding = deserialize_embedding(photo.get("embedding"))
        if image_embedding is None:
            continue
        semantic_score = cosine_similarity(query_embedding, image_embedding)
        ocr_text = (photo.get("ocr_text") or "").lower()
        matched_terms = sum(1 for term in query_terms if term in ocr_text)
        ocr_match = bool(matched_terms)
        text_boost = min(0.15, matched_terms * 0.05)
        combined = semantic_score + text_boost
        photo["semantic_score"] = round(semantic_score, 6)
        photo["ocr_match"] = ocr_match
        photo["similarity_score"] = round(combined, 6)
        results.append(photo)

    results.sort(key=lambda item: item["similarity_score"], reverse=True)
    return results[:limit]
