"""Photo indexing, AI analysis, face detection and duplicate detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.database import get_conn
from app.services.clip_service import ClipServiceError, analyze_image
from app.services.face_service import replace_photo_faces
from app.services.image_service import (
    image_dimensions,
    iter_images,
    perceptual_hash,
    sha256_file,
    simple_category,
)
from app.services.ocr_service import (
    OcrServiceError,
    analyze_document,
    should_run_ocr,
)

logger = logging.getLogger(__name__)


def _categorize_and_embed(
    path: Path,
) -> tuple[str, str, str | None]:
    """
    Classify an image and generate its CLIP embedding.

    Filename-based categorisation is used only if CLIP is unavailable.
    """

    try:
        category, description, embedding = analyze_image(path)

        return category, description, embedding

    except ClipServiceError as exc:
        logger.warning(
            "CLIP analysis failed for %s: %s",
            path,
            exc,
        )

        category, description = simple_category(path)

        return category, description, None

    except Exception as exc:
        logger.exception(
            "Unexpected CLIP analysis error for %s.",
            path,
        )

        category, description = simple_category(path)

        return category, description, None


def _extract_document_metadata(
    path: Path,
    category: str,
) -> tuple[str | None, float | None, str, int]:
    """
    Run OCR and document classification when the image is likely
    to contain meaningful text.
    """

    if not should_run_ocr(path, category):
        return None, None, "non_document", 0

    try:
        text, confidence, document_type, important = (
            analyze_document(path, category)
        )

        return (
            text or None,
            confidence,
            document_type,
            int(important),
        )

    except OcrServiceError as exc:
        logger.warning(
            "OCR failed for %s: %s",
            path,
            exc,
        )

    except Exception:
        logger.exception(
            "Unexpected OCR error for %s.",
            path,
        )

    fallback_type = (
        category
        if category in {
            "receipt",
            "prescription",
            "document",
        }
        else "non_document"
    )

    important = int(
        fallback_type in {
            "prescription",
            "document",
        }
    )

    return (
        None,
        None,
        fallback_type,
        important,
    )


def _get_existing_photo(
    connection: Any,
    resolved_path: str,
) -> dict[str, Any] | None:
    """Return an existing database record for a path."""

    row = connection.execute(
        """
        SELECT *
        FROM photos
        WHERE path = ?
        """,
        (resolved_path,),
    ).fetchone()

    return dict(row) if row else None


def _preserve_existing_ai_values(
    existing: dict[str, Any] | None,
    category: str,
    description: str,
    embedding: str | None,
    ocr_text: str | None,
    ocr_confidence: float | None,
    document_type: str,
    important: int,
) -> tuple[
    str,
    str,
    str | None,
    str | None,
    float | None,
    str,
    int,
]:
    """
    Preserve successful previous AI results when a temporary model
    failure occurs during re-indexing.

    This prevents a valid embedding or OCR result from being replaced
    by NULL.
    """

    if existing is None:
        return (
            category,
            description,
            embedding,
            ocr_text,
            ocr_confidence,
            document_type,
            important,
        )

    existing_embedding = existing.get("embedding")

    if embedding is None and existing_embedding:
        embedding = existing_embedding

        existing_category = existing.get("category")
        existing_description = existing.get("description")

        if existing_category:
            category = str(existing_category)

        if existing_description:
            description = str(existing_description)

    existing_ocr_text = existing.get("ocr_text")

    if ocr_text is None and existing_ocr_text:
        ocr_text = str(existing_ocr_text)
        ocr_confidence = existing.get("ocr_confidence")

        existing_document_type = existing.get(
            "document_type"
        )

        if existing_document_type:
            document_type = str(existing_document_type)

        important = int(
            existing.get(
                "is_important_document",
                important,
            )
            or 0
        )

    return (
        category,
        description,
        embedding,
        ocr_text,
        ocr_confidence,
        document_type,
        important,
    )


def index_folder(
    folder: Path,
) -> dict[str, Any]:
    """
    Index all supported images in a folder.

    Existing records are updated so this function can also be used
    to regenerate missing CLIP embeddings and OCR metadata.
    """

    folder = folder.expanduser().resolve()

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {folder}"
        )

    indexed = 0
    updated = 0
    skipped = 0

    embeddings_created = 0
    embeddings_preserved = 0
    ocr_processed = 0
    faces_processed = 0

    errors: list[dict[str, Any]] = []

    with get_conn() as connection:
        for path in iter_images(folder):
            try:
                resolved_path = str(
                    path.expanduser().resolve()
                )

                existing = _get_existing_photo(
                    connection,
                    resolved_path,
                )

                sha256 = sha256_file(path)
                phash = perceptual_hash(path)
                width, height = image_dimensions(path)

                category, description, embedding = (
                    _categorize_and_embed(path)
                )

                if embedding is not None:
                    embeddings_created += 1
                elif (
                    existing is not None
                    and existing.get("embedding")
                ):
                    embeddings_preserved += 1

                (
                    ocr_text,
                    ocr_confidence,
                    document_type,
                    important,
                ) = _extract_document_metadata(
                    path,
                    category,
                )

                if ocr_text:
                    ocr_processed += 1

                (
                    category,
                    description,
                    embedding,
                    ocr_text,
                    ocr_confidence,
                    document_type,
                    important,
                ) = _preserve_existing_ai_values(
                    existing=existing,
                    category=category,
                    description=description,
                    embedding=embedding,
                    ocr_text=ocr_text,
                    ocr_confidence=ocr_confidence,
                    document_type=document_type,
                    important=important,
                )

                if document_type in {
                    "receipt",
                    "prescription",
                    "document",
                }:
                    category = document_type

                    description = (
                        f"Image classified as {category} "
                        "using CLIP with OCR refinement."
                    )

                connection.execute(
                    """
                    INSERT INTO photos (
                        path,
                        filename,
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
                        is_important_document
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
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
                        is_important_document =
                            excluded.is_important_document
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

                photo_row = connection.execute(
                    """
                    SELECT id
                    FROM photos
                    WHERE path = ?
                    """,
                    (resolved_path,),
                ).fetchone()

                if photo_row:
                    try:
                        replace_photo_faces(
                            connection,
                            int(photo_row["id"]),
                            path,
                        )

                        faces_processed += 1

                    except Exception as face_exc:
                        logger.exception(
                            "Face detection failed for %s.",
                            path,
                        )

                        errors.append(
                            {
                                "path": str(path),
                                "stage": "face_detection",
                                "error": str(face_exc),
                            }
                        )

                if existing is None:
                    indexed += 1
                else:
                    updated += 1

            except Exception as exc:
                logger.exception(
                    "Indexing failed for %s.",
                    path,
                )

                errors.append(
                    {
                        "path": str(path),
                        "stage": "indexing",
                        "error": str(exc),
                    }
                )

    return {
        "folder": str(folder),
        "indexed": indexed,
        "updated": updated,
        "skipped": skipped,
        "embeddings_created": embeddings_created,
        "embeddings_preserved": embeddings_preserved,
        "ocr_processed": ocr_processed,
        "faces_processed": faces_processed,
        "errors": errors,
    }


def list_duplicates(
    max_distance: int = 5,
) -> dict[str, list[list[dict[str, Any]]]]:
    """
    Return exact duplicate and perceptually similar image groups.
    """

    with get_conn() as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM photos
                ORDER BY id
                """
            )
        ]

    exact_by_hash: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        sha256 = row.get("sha256")

        if not sha256:
            continue

        exact_by_hash.setdefault(
            str(sha256),
            [],
        ).append(row)

    exact_groups = [
        group
        for group in exact_by_hash.values()
        if len(group) > 1
    ]

    near_groups: list[list[dict[str, Any]]] = []
    used_photo_ids: set[int] = set()

    for index, first in enumerate(rows):
        first_id = int(first["id"])
        first_phash = first.get("phash")

        if (
            first_id in used_photo_ids
            or not first_phash
        ):
            continue

        try:
            first_hash_value = int(
                str(first_phash),
                16,
            )

        except ValueError:
            logger.warning(
                "Ignoring invalid perceptual hash for photo %s.",
                first_id,
            )
            continue

        group = [first]

        for second in rows[index + 1 :]:
            second_id = int(second["id"])
            second_phash = second.get("phash")

            if (
                second_id in used_photo_ids
                or not second_phash
            ):
                continue

            if (
                first.get("sha256")
                == second.get("sha256")
            ):
                continue

            try:
                second_hash_value = int(
                    str(second_phash),
                    16,
                )

            except ValueError:
                logger.warning(
                    "Ignoring invalid perceptual hash for photo %s.",
                    second_id,
                )
                continue

            distance = (
                first_hash_value
                ^ second_hash_value
            ).bit_count()

            if distance <= max_distance:
                group.append(second)

        if len(group) > 1:
            near_groups.append(group)

            used_photo_ids.update(
                int(item["id"])
                for item in group
            )

    return {
        "exact_duplicates": exact_groups,
        "near_duplicates": near_groups,
    }