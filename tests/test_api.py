import sqlite3
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.core.database import get_conn, init_db
from app.main import app
from app.services.clip_service import ClipServiceError
from app.services.indexer import index_folder, list_duplicates

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_index_folder():
    response = client.post("/index", json={"folder": "/folder/that/does/not/exist"})
    assert response.status_code == 400


def test_search_returns_503_when_clip_unavailable():
    with patch("app.main.semantic_search", side_effect=ClipServiceError("model unavailable")):
        response = client.get("/search?q=beach")
    assert response.status_code == 503
    assert "CLIP search unavailable" in response.json()["detail"]


def test_index_falls_back_when_clip_unavailable(tmp_path, monkeypatch):
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    image_path = photo_dir / "receipt_test.jpg"
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(image_path, format="JPEG")

    test_db = tmp_path / "test.db"
    monkeypatch.setattr("app.core.config.DB_PATH", test_db)
    monkeypatch.setattr("app.core.config.DATA_DIR", tmp_path)
    init_db()

    with patch("app.services.indexer.analyze_image", side_effect=ClipServiceError("model unavailable")):
        result = index_folder(photo_dir)

    assert result["indexed"] == 1
    assert result["errors"] == []

    with get_conn() as conn:
        row = conn.execute("SELECT category, embedding FROM photos WHERE filename = ?", ("receipt_test.jpg",)).fetchone()

    assert row["category"] == "receipt"
    assert row["embedding"] is None


def test_duplicate_detection_preserved(tmp_path, monkeypatch):
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    test_db = tmp_path / "test.db"
    monkeypatch.setattr("app.core.config.DB_PATH", test_db)
    monkeypatch.setattr("app.core.config.DATA_DIR", tmp_path)
    init_db()

    exact_bytes = b"exact-duplicate-image-bytes"
    near_a_bytes = b"near-duplicate-image-a-bytes"
    near_b_bytes = b"near-duplicate-image-b-bytes"

    (photo_dir / "exact_one.jpg").write_bytes(exact_bytes)
    (photo_dir / "exact_two.jpg").write_bytes(exact_bytes)
    (photo_dir / "near_one.jpg").write_bytes(near_a_bytes)
    (photo_dir / "near_two.jpg").write_bytes(near_b_bytes)

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO photos (path, filename, sha256, phash, category, description, width, height, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(photo_dir / "exact_one.jpg"), "exact_one.jpg", "sha-exact", "ff00ff00ff00ff00", "other", "test", 100, 100, None),
        )
        conn.execute(
            """
            INSERT INTO photos (path, filename, sha256, phash, category, description, width, height, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(photo_dir / "exact_two.jpg"), "exact_two.jpg", "sha-exact", "ff00ff00ff00ff00", "other", "test", 100, 100, None),
        )
        conn.execute(
            """
            INSERT INTO photos (path, filename, sha256, phash, category, description, width, height, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(photo_dir / "near_one.jpg"), "near_one.jpg", "sha-near-a", "0000000000000001", "other", "test", 100, 100, None),
        )
        conn.execute(
            """
            INSERT INTO photos (path, filename, sha256, phash, category, description, width, height, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(photo_dir / "near_two.jpg"), "near_two.jpg", "sha-near-b", "0000000000000003", "other", "test", 100, 100, None),
        )

    result = list_duplicates(max_distance=5)
    assert len(result["exact_duplicates"]) == 1
    assert len(result["exact_duplicates"][0]) == 2
    assert len(result["near_duplicates"]) == 1
    assert len(result["near_duplicates"][0]) == 2


def test_database_migration_adds_embedding_column(tmp_path, monkeypatch):
    legacy_db = tmp_path / "legacy.db"
    monkeypatch.setattr("app.core.config.DB_PATH", legacy_db)
    monkeypatch.setattr("app.core.config.DATA_DIR", tmp_path)

    with sqlite3.connect(legacy_db) as conn:
        conn.executescript(
            """
            CREATE TABLE photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                phash TEXT,
                category TEXT,
                description TEXT,
                width INTEGER,
                height INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

    init_db()

    with sqlite3.connect(legacy_db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}

    assert "embedding" in columns
