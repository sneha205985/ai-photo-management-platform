from pathlib import Path

from app.core.database import get_conn
from app.services.clip_service import ClipServiceError, analyze_image
from app.services.image_service import (
    image_dimensions,
    iter_images,
    perceptual_hash,
    sha256_file,
    simple_category,
)
from app.services.ocr_service import OcrServiceError, analyze_document, should_run_ocr
from app.services.face_service import replace_photo_faces


def _categorize_and_embed(path: Path) -> tuple[str, str, str | None]:
    try:
        return analyze_image(path)
    except (ClipServiceError, Exception):
        category, description = simple_category(path)
        return category, description, None


def _extract_document_metadata(
    path: Path, category: str
) -> tuple[str | None, float | None, str, int]:
    if not should_run_ocr(path, category):
        return None, None, "non_document", 0
    try:
        text, confidence, document_type, important = analyze_document(path, category)
        return text or None, confidence, document_type, int(important)
    except (OcrServiceError, Exception):
        fallback_type = category if category in {"receipt", "prescription", "document"} else "non_document"
        return None, None, fallback_type, int(fallback_type in {"prescription", "document"})


def index_folder(folder: Path) -> dict:
    indexed = 0
    updated = 0
    skipped = 0
    errors: list[dict] = []

    with get_conn() as conn:
        for path in iter_images(folder):
            try:
                sha256 = sha256_file(path)
                phash = perceptual_hash(path)
                width, height = image_dimensions(path)
                category, description, embedding = _categorize_and_embed(path)
                ocr_text, ocr_confidence, document_type, important = _extract_document_metadata(
                    path, category
                )
                if document_type in {"receipt", "prescription", "document"}:
                    category = document_type
                    description = f"Image classified as {category} using CLIP with OCR refinement."

                resolved_path = str(path.resolve())
                existing = conn.execute(
                    "SELECT id FROM photos WHERE path = ?",
                    (resolved_path,),
                ).fetchone()

                conn.execute(
                    """
                    INSERT INTO photos
                    (path, filename, sha256, phash, category, description, width, height,
                     embedding, ocr_text, document_type, ocr_confidence, is_important_document)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        filename = excluded.filename,
                        sha256 = excluded.sha256,
                        phash = excluded.phash,
                        category = excluded.category,
                        description = excluded.description,
                        width = excluded.width,
                        height = excluded.height,
                        embedding = excluded.embedding,
                        ocr_text = excluded.ocr_text,
                        document_type = excluded.document_type,
                        ocr_confidence = excluded.ocr_confidence,
                        is_important_document = excluded.is_important_document
                    """,
                    (
                        resolved_path,
                        path.name,
                        sha256,
                        phash,
                        category,
                        description,
                        width,
                        height,
                        embedding,
                        ocr_text,
                        document_type,
                        ocr_confidence,
                        important,
                    ),
                )

                photo_row = conn.execute(
                    "SELECT id FROM photos WHERE path = ?",
                    (resolved_path,),
                ).fetchone()
                if photo_row:
                    try:
                        replace_photo_faces(conn, int(photo_row["id"]), path)
                    except Exception as face_exc:
                        errors.append({"path": str(path), "stage": "face_detection", "error": str(face_exc)})

                if existing:
                    updated += 1
                else:
                    indexed += 1
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})

    return {"indexed": indexed, "updated": updated, "skipped": skipped, "errors": errors}


def list_duplicates(max_distance: int = 5) -> dict:
    with get_conn() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM photos ORDER BY id")]

    exact: dict[str, list[dict]] = {}
    for row in rows:
        exact.setdefault(row["sha256"], []).append(row)
    exact_groups = [group for group in exact.values() if len(group) > 1]

    near_groups: list[list[dict]] = []
    used: set[int] = set()
    for i, first in enumerate(rows):
        if first["id"] in used or not first["phash"]:
            continue
        group = [first]
        for second in rows[i + 1 :]:
            if second["id"] in used or not second["phash"]:
                continue
            distance = bin(int(first["phash"], 16) ^ int(second["phash"], 16)).count("1")
            if distance <= max_distance and first["sha256"] != second["sha256"]:
                group.append(second)
        if len(group) > 1:
            near_groups.append(group)
            used.update(item["id"] for item in group)

    return {"exact_duplicates": exact_groups, "near_duplicates": near_groups}
