import pytest

from app.services.clip_service import (
    ClipService,
    ClipServiceError,
    cosine_similarity,
    deserialize_embedding,
    semantic_search,
    serialize_embedding,
)


def test_embedding_serialization_round_trip():
    embedding = [0.12, -0.34, 0.56, 0.0, 1.0]
    raw = serialize_embedding(embedding)
    assert isinstance(raw, str)
    assert deserialize_embedding(raw) == embedding


def test_deserialize_embedding_handles_missing_values():
    assert deserialize_embedding(None) is None
    assert deserialize_embedding("") is None


def test_cosine_similarity_identical_vectors():
    vector = [1.0, 2.0, 3.0]
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_ordering():
    query = [1.0, 0.0, 0.0]
    closer = [0.95, 0.05, 0.0]
    farther = [0.1, 0.95, 0.0]

    closer_score = cosine_similarity(query, closer)
    farther_score = cosine_similarity(query, farther)
    assert closer_score > farther_score


def test_semantic_search_orders_by_similarity(monkeypatch):
    rows = [
        {"id": 1, "path": "/a.jpg", "filename": "a.jpg", "embedding": serialize_embedding([1.0, 0.0, 0.0])},
        {"id": 2, "path": "/b.jpg", "filename": "b.jpg", "embedding": serialize_embedding([0.7, 0.7, 0.0])},
        {"id": 3, "path": "/c.jpg", "filename": "c.jpg", "embedding": serialize_embedding([0.0, 1.0, 0.0])},
        {"id": 4, "path": "/d.jpg", "filename": "d.jpg", "embedding": None},
    ]

    class FakeCursor:
        def fetchall(self):
            return rows

    class FakeConn:
        def execute(self, *_args, **_kwargs):
            return FakeCursor()

        def commit(self):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeClip:
        def encode_text(self, _text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr("app.services.clip_service.get_conn", lambda: FakeConn())
    monkeypatch.setattr("app.services.clip_service.get_clip_service", lambda: FakeClip())

    results = semantic_search("beach vacation", limit=2)
    assert len(results) == 2
    assert results[0]["filename"] == "a.jpg"
    assert results[1]["filename"] == "b.jpg"
    assert results[0]["similarity_score"] >= results[1]["similarity_score"]
    assert all("similarity_score" in item for item in results)


def test_graceful_model_failure_during_search(monkeypatch):
    class BrokenClip:
        def encode_text(self, _text: str) -> list[float]:
            raise ClipServiceError("model unavailable")

    monkeypatch.setattr("app.services.clip_service.get_clip_service", lambda: BrokenClip())

    with pytest.raises(ClipServiceError):
        semantic_search("dog on grass")


def test_clip_service_singleton():
    first = ClipService()
    second = ClipService()
    assert first is second
