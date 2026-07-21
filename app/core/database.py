import sqlite3
from contextlib import contextmanager
from app.core.config import DB_PATH, DATA_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
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
CREATE INDEX IF NOT EXISTS idx_photos_sha256 ON photos(sha256);
CREATE INDEX IF NOT EXISTS idx_photos_category ON photos(category);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply additive schema changes for databases created before CLIP support."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}
    additions = {
        "embedding": "TEXT",
        "ocr_text": "TEXT",
        "document_type": "TEXT",
        "ocr_confidence": "REAL",
        "is_important_document": "INTEGER DEFAULT 0",
        "face_count": "INTEGER DEFAULT 0",
    }
    for name, sql_type in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE photos ADD COLUMN {name} {sql_type}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_document_type ON photos(document_type)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_photos_important_document "
        "ON photos(is_important_document)"
    )

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            person_id INTEGER,
            bbox_x INTEGER NOT NULL,
            bbox_y INTEGER NOT NULL,
            bbox_width INTEGER NOT NULL,
            bbox_height INTEGER NOT NULL,
            confidence REAL,
            embedding TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE,
            FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
        CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);
        """
    )


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        _migrate_schema(conn)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
