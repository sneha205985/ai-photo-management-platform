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
    "document": (
        "a photo of a document, form, certificate, "
        "or official paperwork"
    ),
    "prescription": (
        "a photo of a medical prescription, medicine bottle, "
        "or pharmacy label"
    ),
    "receipt": (
        "a photo of a store receipt, restaurant bill, "
        "or purchase invoice"
    ),
    "people": (
        "a photo of a person, portrait, selfie, "
        "or group of people"
    ),
    "travel": (
        "a photo of travel, vacation, landmarks, beaches, "
        "or scenic destinations"
    ),
    "pet": (
        "a photo of a pet such as a dog, cat, puppy, or kitten"
    ),
    "other": (
        "a photo of miscellaneous objects or an uncategorized scene"
    ),
}


class ClipServiceError(Exception):
    """Raised when the CLIP model is unavailable or inference fails."""


class ClipService:
    """Lazy-loaded singleton wrapper around the CLIP model."""

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
        return (
            self._model is not None
            and self._processor is not None
        )

    def _ensure_loaded(self) -> None:
        if self.available:
            return

        if self._load_error is not None:
            raise ClipServiceError(self._load_error)

        with self._model_lock:
            if self.available:
                return

            if self._load_error is not None:
                raise ClipServiceError(self._load_error)

            try:
                from transformers import CLIPModel, CLIPProcessor

                logger.info(
                    "Loading CLIP model %s. "
                    "The first run may download model weights.",
                    MODEL_ID,
                )

                self._processor = CLIPProcessor.from_pretrained(
                    MODEL_ID
                )

                self._model = CLIPModel.from_pretrained(
                    MODEL_ID
                )

                self._model.eval()

                logger.info(
                    "CLIP model loaded successfully."
                )

            except Exception as exc:
                self._load_error = str(exc)

                logger.exception(
                    "Failed to load CLIP model."
                )

                raise ClipServiceError(
                    f"Could not load CLIP model: {exc}"
                ) from exc

    @staticmethod
    def _normalise_tensor(features: Any) -> Any:
        """L2-normalise a PyTorch feature tensor."""

        norm = features.norm(
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-12)

        return features / norm

    def encode_image(self, path: Path) -> list[float]:
        """Generate a normalised CLIP embedding for one image."""

        self._ensure_loaded()

        import torch

        try:
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")

            inputs = self._processor(
                images=rgb_image,
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"]

            with torch.no_grad():
                vision_output = self._model.vision_model(
                    pixel_values=pixel_values,
                    return_dict=True,
                )

                pooled_output = vision_output.pooler_output

                features = self._model.visual_projection(
                    pooled_output
                )

                features = self._normalise_tensor(
                    features
                )

            return (
                features
                .squeeze(0)
                .detach()
                .cpu()
                .tolist()
            )

        except ClipServiceError:
            raise

        except Exception as exc:
            logger.exception(
                "CLIP image encoding failed for %s.",
                path,
            )

            raise ClipServiceError(
                f"Could not generate image embedding for "
                f"{path.name}: {exc}"
            ) from exc

    def encode_text(self, text: str) -> list[float]:
        """Generate a normalised CLIP embedding for text."""

        self._ensure_loaded()

        import torch

        clean_text = text.strip()

        if not clean_text:
            raise ClipServiceError(
                "Search text cannot be empty."
            )

        try:
            inputs = self._processor(
                text=[clean_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            with torch.no_grad():
                text_output = self._model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get(
                        "attention_mask"
                    ),
                    return_dict=True,
                )

                pooled_output = text_output.pooler_output

                features = self._model.text_projection(
                    pooled_output
                )

                features = self._normalise_tensor(
                    features
                )

            return (
                features
                .squeeze(0)
                .detach()
                .cpu()
                .tolist()
            )

        except ClipServiceError:
            raise

        except Exception as exc:
            logger.exception(
                "CLIP text encoding failed."
            )

            raise ClipServiceError(
                f"Could not generate text embedding: {exc}"
            ) from exc

    def classify_image(
        self,
        path: Path,
    ) -> tuple[str, str]:
        """Classify an image using zero-shot CLIP prompts."""

        self._ensure_loaded()

        import torch

        try:
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")

            labels = list(
                CATEGORY_PROMPTS.keys()
            )

            prompts = [
                CATEGORY_PROMPTS[label]
                for label in labels
            ]

            inputs = self._processor(
                text=prompts,
                images=rgb_image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            with torch.no_grad():
                outputs = self._model(
                    **inputs
                )

                logits = (
                    outputs
                    .logits_per_image
                    .squeeze(0)
                )

                probabilities = logits.softmax(
                    dim=-1
                )

                best_index = int(
                    probabilities.argmax().item()
                )

                confidence = float(
                    probabilities[best_index].item()
                )

            category = labels[best_index]

            description = (
                f"Image classified as {category} using "
                f"CLIP zero-shot classification "
                f"(confidence {confidence:.3f})."
            )

            return category, description

        except ClipServiceError:
            raise

        except Exception as exc:
            logger.exception(
                "CLIP classification failed for %s.",
                path,
            )

            raise ClipServiceError(
                f"Could not classify {path.name}: {exc}"
            ) from exc


def get_clip_service() -> ClipService:
    """Return the shared CLIP service instance."""

    return ClipService()


def serialize_embedding(
    embedding: list[float],
) -> str:
    """Convert an embedding to JSON for SQLite storage."""

    return json.dumps(
        embedding,
        separators=(",", ":"),
    )


def deserialize_embedding(
    raw: str | None,
) -> list[float] | None:
    """Convert a stored JSON embedding back to a list."""

    if not raw:
        return None

    try:
        embedding = json.loads(raw)

    except (TypeError, json.JSONDecodeError):
        logger.warning(
            "Ignoring an invalid stored embedding."
        )
        return None

    if not isinstance(embedding, list):
        return None

    try:
        return [
            float(value)
            for value in embedding
        ]

    except (TypeError, ValueError):
        return None


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    """Return cosine similarity between two vectors."""

    if not left or not right:
        return 0.0

    if len(left) != len(right):
        logger.warning(
            "Embedding size mismatch: %s and %s.",
            len(left),
            len(right),
        )
        return 0.0

    dot_product = sum(
        left_value * right_value
        for left_value, right_value
        in zip(left, right)
    )

    left_norm = sum(
        value * value
        for value in left
    ) ** 0.5

    right_norm = sum(
        value * value
        for value in right
    ) ** 0.5

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (
        left_norm * right_norm
    )


def analyze_image(
    path: Path,
) -> tuple[str, str, str]:
    """
    Return the CLIP category, description and serialized
    embedding for an image.
    """

    clip_service = get_clip_service()

    category, description = (
        clip_service.classify_image(path)
    )

    embedding = clip_service.encode_image(
        path
    )

    return (
        category,
        description,
        serialize_embedding(embedding),
    )


def semantic_search(
    query: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Search indexed images using CLIP similarity and OCR text.

    Records without valid embeddings are ignored safely.
    """

    from app.core.database import get_conn

    clean_query = query.strip()

    if not clean_query:
        return []

    clip_service = get_clip_service()

    query_embedding = clip_service.encode_text(
        clean_query
    )

    query_terms = {
        term
        for term in clean_query.lower().split()
        if len(term) >= 3
    }

    with get_conn() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM photos
            WHERE embedding IS NOT NULL
              AND TRIM(embedding) != ''
            """
        ).fetchall()

    results: list[dict[str, Any]] = []

    for row in rows:
        photo = dict(row)

        image_embedding = deserialize_embedding(
            photo.get("embedding")
        )

        if image_embedding is None:
            continue

        semantic_score = cosine_similarity(
            query_embedding,
            image_embedding,
        )

        ocr_text = (
            photo.get("ocr_text")
            or ""
        ).lower()

        matched_terms = sum(
            1
            for term in query_terms
            if term in ocr_text
        )

        text_boost = min(
            0.15,
            matched_terms * 0.05,
        )

        combined_score = (
            semantic_score + text_boost
        )

        photo["semantic_score"] = round(
            semantic_score,
            6,
        )

        photo["ocr_match"] = (
            matched_terms > 0
        )

        photo["similarity_score"] = round(
            combined_score,
            6,
        )

        # Avoid returning the large raw embedding in API results.
        photo.pop(
            "embedding",
            None,
        )

        results.append(photo)

    results.sort(
        key=lambda item: item[
            "similarity_score"
        ],
        reverse=True,
    )

    return results[:limit]