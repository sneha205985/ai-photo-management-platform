"""OCR and document-type refinement utilities.

EasyOCR is loaded lazily so the API can still start when the optional OCR
runtime is unavailable. Indexing falls back gracefully on OCR errors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

RECEIPT_KEYWORDS = {
    "total", "subtotal", "tax", "amount", "invoice", "payment", "cash",
    "card", "balance", "receipt", "change", "qty", "price",
}
PRESCRIPTION_KEYWORDS = {
    "patient", "doctor", "dosage", "tablet", "medicine", "pharmacy",
    "prescription", "mg", "daily", "capsule", "dose", "rx",
}
DOCUMENT_KEYWORDS = {
    "certificate", "application", "form", "passport", "identification",
    "address", "date", "signature", "document", "name", "issued",
}


class OcrServiceError(Exception):
    """Raised when OCR is unavailable or inference fails."""


class OcrService:
    """Lazy-loaded singleton wrapper around EasyOCR."""

    _instance: "OcrService | None" = None
    _instance_lock = Lock()

    def __new__(cls) -> "OcrService":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._reader: Any = None
        self._load_error: str | None = None
        self._reader_lock = Lock()
        self._initialized = True

    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return
        if self._load_error:
            raise OcrServiceError(self._load_error)

        with self._reader_lock:
            if self._reader is not None:
                return
            try:
                import easyocr

                logger.info("Loading EasyOCR reader (first run may download model files)...")
                self._reader = easyocr.Reader(["en"], gpu=False)
            except Exception as exc:  # pragma: no cover - environment dependent
                self._load_error = str(exc)
                logger.warning("EasyOCR unavailable: %s", exc)
                raise OcrServiceError(str(exc)) from exc

    def extract_text(self, path: Path) -> tuple[str, float | None]:
        """Return combined text and mean confidence for an image."""
        self._ensure_loaded()
        try:
            results = self._reader.readtext(str(path), detail=1, paragraph=False)
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            raise OcrServiceError(str(exc)) from exc

        texts: list[str] = []
        confidences: list[float] = []
        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            text = str(item[1]).strip()
            if text:
                texts.append(text)
                try:
                    confidences.append(float(item[2]))
                except (TypeError, ValueError):
                    pass

        confidence = sum(confidences) / len(confidences) if confidences else None
        return "\n".join(texts), confidence


def get_ocr_service() -> OcrService:
    return OcrService()


def _keyword_hits(text: str, keywords: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def classify_document_type(ocr_text: str, clip_category: str | None) -> str:
    """Refine a CLIP category using OCR keyword evidence."""
    receipt_hits = _keyword_hits(ocr_text, RECEIPT_KEYWORDS)
    prescription_hits = _keyword_hits(ocr_text, PRESCRIPTION_KEYWORDS)
    document_hits = _keyword_hits(ocr_text, DOCUMENT_KEYWORDS)

    scores = {
        "receipt": receipt_hits,
        "prescription": prescription_hits,
        "document": document_hits,
    }
    best_type = max(scores, key=scores.get)
    if scores[best_type] >= 2:
        return best_type

    if clip_category in {"receipt", "prescription", "document"}:
        return clip_category
    if ocr_text.strip() and max(scores.values()) >= 1:
        return best_type
    return "non_document"


def is_important_document(document_type: str, ocr_text: str) -> bool:
    """Flag documents likely to deserve prominent treatment."""
    if document_type in {"prescription", "document"}:
        return True
    if document_type == "receipt":
        return any(token in ocr_text.lower() for token in {"total", "invoice", "amount"})
    return False


def should_run_ocr(path: Path, clip_category: str | None) -> bool:
    """Cheap routing rule to avoid OCR on obviously non-document photos."""
    if clip_category in {"receipt", "prescription", "document"}:
        return True
    name = path.name.lower()
    hints = {
        "receipt", "invoice", "bill", "prescription", "medicine", "medical",
        "document", "passport", "certificate", "form", "letter", "scan",
    }
    return any(hint in name for hint in hints)


def analyze_document(path: Path, clip_category: str | None) -> tuple[str, float | None, str, bool]:
    """Run OCR and return text, confidence, document type, importance flag."""
    text, confidence = get_ocr_service().extract_text(path)
    document_type = classify_document_type(text, clip_category)
    important = is_important_document(document_type, text)
    return text, confidence, document_type, important
