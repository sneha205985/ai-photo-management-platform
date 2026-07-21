from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.core.config import PHOTO_DIR
from app.core.database import get_conn, init_db
from app.services.clip_service import ClipServiceError, semantic_search
from app.services.indexer import index_folder, list_duplicates
from app.services.face_service import rebuild_people
from app.api.google_photos import router as google_photos_router

app = FastAPI(title="AI Photo Management Platform", version="0.3.0")
app.include_router(google_photos_router)


class IndexRequest(BaseModel):
    folder: str | None = None


class PersonRenameRequest(BaseModel):
    label: str


@app.on_event("startup")
def startup() -> None:
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/index")
def index_photos(request: IndexRequest) -> dict:
    folder = Path(request.folder) if request.folder else PHOTO_DIR
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="Folder does not exist or is not a directory")
    return index_folder(folder)


@app.get("/photos")
def photos(
    category: str | None = None,
    document_type: str | None = None,
    important_only: bool = False,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    clauses: list[str] = []
    params: list[object] = []
    if category:
        clauses.append("category = ?")
        params.append(category)
    if document_type:
        clauses.append("document_type = ?")
        params.append(document_type)
    if important_only:
        clauses.append("is_important_document = 1")

    query = "SELECT * FROM photos"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_conn() as conn:
        return [dict(row) for row in conn.execute(query, tuple(params))]


@app.get("/documents")
def documents(
    document_type: str | None = None,
    important_only: bool = False,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    clauses = ["document_type IN ('receipt', 'prescription', 'document')"]
    params: list[object] = []
    if document_type:
        clauses.append("document_type = ?")
        params.append(document_type)
    if important_only:
        clauses.append("is_important_document = 1")
    query = (
        "SELECT * FROM photos WHERE " + " AND ".join(clauses) +
        " ORDER BY id DESC LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    with get_conn() as conn:
        return [dict(row) for row in conn.execute(query, tuple(params))]


@app.get("/duplicates")
def duplicates(max_distance: int = Query(5, ge=0, le=20)) -> dict:
    return list_duplicates(max_distance=max_distance)


@app.get("/search")
def search(q: str, limit: int = Query(50, ge=1, le=200)) -> list[dict]:
    try:
        return semantic_search(q, limit=limit)
    except ClipServiceError as exc:
        raise HTTPException(status_code=503, detail=f"CLIP search unavailable: {exc}") from exc


@app.post("/faces/rebuild")
def rebuild_face_groups(
    similarity_threshold: float = Query(0.82, ge=0.5, le=0.99),
) -> dict:
    return rebuild_people(threshold=similarity_threshold)


@app.get("/people")
def people(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    query = """
        SELECT p.id, p.label, p.created_at, COUNT(f.id) AS face_count,
               COUNT(DISTINCT f.photo_id) AS photo_count
        FROM people p
        LEFT JOIN faces f ON f.person_id = p.id
        GROUP BY p.id
        ORDER BY photo_count DESC, p.id
        LIMIT ? OFFSET ?
    """
    with get_conn() as conn:
        return [dict(row) for row in conn.execute(query, (limit, offset))]


@app.get("/people/{person_id}/photos")
def person_photos(
    person_id: int,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    query = """
        SELECT DISTINCT photos.*
        FROM photos
        JOIN faces ON faces.photo_id = photos.id
        WHERE faces.person_id = ?
        ORDER BY photos.id DESC
        LIMIT ? OFFSET ?
    """
    with get_conn() as conn:
        exists = conn.execute("SELECT id FROM people WHERE id = ?", (person_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Person group not found")
        return [dict(row) for row in conn.execute(query, (person_id, limit, offset))]


@app.patch("/people/{person_id}")
def rename_person(person_id: int, request: PersonRenameRequest) -> dict:
    label = request.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label cannot be empty")
    with get_conn() as conn:
        cursor = conn.execute("UPDATE people SET label = ? WHERE id = ?", (label, person_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Person group not found")
    return {"id": person_id, "label": label}
