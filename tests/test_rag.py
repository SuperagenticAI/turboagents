import asyncio
import importlib
from pathlib import Path
import tempfile

import numpy as np

from turboagents.quant import quantize
from turboagents.rag import (
    TurboChroma,
    TurboFAISS,
    TurboLanceDB,
    TurboPgvector,
    TurboSurrealDB,
)


def test_turbo_faiss_search_returns_hits() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((32, 64), dtype=np.float32)
    query = vectors[0].copy()

    index = TurboFAISS(dim=64, bits=3.5, seed=0)
    index.add(vectors)
    results = index.search(query, k=5, rerank_top=10)

    assert len(results) == 5
    assert results[0]["index"] == 0


def test_turbo_chroma_search_returns_hits(monkeypatch) -> None:
    collections_by_path: dict[str | None, dict[str, object]] = {}

    class FakeCollection:
        def __init__(self) -> None:
            self.rows: list[tuple[str, np.ndarray, object]] = []

        def add(self, ids, embeddings, metadatas):
            for row_id, embedding, metadata in zip(ids, embeddings, metadatas):
                self.rows.append((row_id, np.asarray(embedding, dtype=np.float32), metadata))

        def query(self, query_embeddings, n_results, include):
            query = np.asarray(query_embeddings[0], dtype=np.float32)
            scored = []
            for row_id, embedding, metadata in self.rows:
                score = float(np.dot(query, embedding))
                scored.append((score, row_id, metadata))
            scored.sort(key=lambda item: item[0], reverse=True)
            top = scored[:n_results]
            return {
                "ids": [[row_id for _score, row_id, _metadata in top]],
                "metadatas": [[metadata for _score, _row_id, metadata in top]],
                "distances": [[-score for score, _row_id, _metadata in top]],
            }

        def count(self):
            return len(self.rows)

    class FakeClient:
        def __init__(self, path=None) -> None:
            self.path = path
            self.collections = collections_by_path.setdefault(path, {})

        def get_or_create_collection(self, name, metadata=None):
            collection = self.collections.get(name)
            if collection is None:
                collection = FakeCollection()
                self.collections[name] = collection
            return collection

        def get_collection(self, name):
            return self.collections[name]

        def delete_collection(self, name):
            self.collections.pop(name, None)

    class FakeChromaModule:
        @staticmethod
        def Client():
            return FakeClient(path=None)

        @staticmethod
        def PersistentClient(path):
            return FakeClient(path=path)

    monkeypatch.setattr("turboagents.rag.chroma._chroma_available", lambda: True)
    monkeypatch.setattr(importlib, "import_module", lambda name: FakeChromaModule)

    rng = np.random.default_rng(2)
    vectors = rng.standard_normal((16, 64), dtype=np.float32)
    query = vectors[4].copy()

    index = TurboChroma(path="/tmp/fake-chroma", dim=64, bits=3.5, seed=2)
    index.create_collection("docs", vectors)
    results = index.search(query, k=3, rerank_top=8)

    assert len(results) == 3
    assert results[0]["index"] == 4


def test_turbo_chroma_reopened_collection_falls_back_without_rerank_state(monkeypatch) -> None:
    collections_by_path: dict[str | None, dict[str, object]] = {}

    class FakeCollection:
        def __init__(self) -> None:
            self.rows: list[tuple[str, np.ndarray, object]] = []

        def add(self, ids, embeddings, metadatas):
            for row_id, embedding, metadata in zip(ids, embeddings, metadatas):
                self.rows.append((row_id, np.asarray(embedding, dtype=np.float32), metadata))

        def query(self, query_embeddings, n_results, include):
            query = np.asarray(query_embeddings[0], dtype=np.float32)
            scored = []
            for row_id, embedding, metadata in self.rows:
                score = float(np.dot(query, embedding))
                scored.append((score, row_id, metadata))
            scored.sort(key=lambda item: item[0], reverse=True)
            top = scored[:n_results]
            return {
                "ids": [[row_id for _score, row_id, _metadata in top]],
                "metadatas": [[metadata for _score, _row_id, metadata in top]],
                "distances": [[-score for score, _row_id, _metadata in top]],
            }

        def count(self):
            return len(self.rows)

    class FakeClient:
        def __init__(self, path=None) -> None:
            self.path = path
            self.collections = collections_by_path.setdefault(path, {})

        def get_or_create_collection(self, name, metadata=None):
            collection = self.collections.get(name)
            if collection is None:
                collection = FakeCollection()
                self.collections[name] = collection
            return collection

        def get_collection(self, name):
            return self.collections[name]

        def delete_collection(self, name):
            self.collections.pop(name, None)

    class FakeChromaModule:
        @staticmethod
        def Client():
            return FakeClient(path=None)

        @staticmethod
        def PersistentClient(path):
            return FakeClient(path=path)

    monkeypatch.setattr("turboagents.rag.chroma._chroma_available", lambda: True)
    monkeypatch.setattr(importlib, "import_module", lambda name: FakeChromaModule)

    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((10, 64), dtype=np.float32)
    query = vectors[6].copy()

    writer = TurboChroma(path="/tmp/fake-chroma-reopen", dim=64, bits=3.5, seed=3)
    writer.create_collection("docs", vectors)

    reopened = TurboChroma(path="/tmp/fake-chroma-reopen", dim=64, bits=3.5, seed=3)
    reopened.open_collection("docs")
    results = reopened.search(query, k=3, rerank_top=8)

    assert len(results) == 3


def test_turbo_chroma_create_collection_replaces_existing_rows(monkeypatch) -> None:
    collections_by_path: dict[str | None, dict[str, object]] = {}

    class FakeCollection:
        def __init__(self) -> None:
            self.rows: list[tuple[str, np.ndarray, object]] = []

        def add(self, ids, embeddings, metadatas):
            for row_id, embedding, metadata in zip(ids, embeddings, metadatas):
                self.rows.append((row_id, np.asarray(embedding, dtype=np.float32), metadata))

        def query(self, query_embeddings, n_results, include):
            query = np.asarray(query_embeddings[0], dtype=np.float32)
            scored = []
            for row_id, embedding, metadata in self.rows:
                score = float(np.dot(query, embedding))
                scored.append((score, row_id, metadata))
            scored.sort(key=lambda item: item[0], reverse=True)
            top = scored[:n_results]
            return {
                "ids": [[row_id for _score, row_id, _metadata in top]],
                "metadatas": [[metadata for _score, _row_id, metadata in top]],
                "distances": [[-score for score, _row_id, _metadata in top]],
            }

        def count(self):
            return len(self.rows)

    class FakeClient:
        def __init__(self, path=None) -> None:
            self.path = path
            self.collections = collections_by_path.setdefault(path, {})

        def get_or_create_collection(self, name, metadata=None):
            collection = self.collections.get(name)
            if collection is None:
                collection = FakeCollection()
                self.collections[name] = collection
            return collection

        def get_collection(self, name):
            return self.collections[name]

        def delete_collection(self, name):
            self.collections.pop(name, None)

    class FakeChromaModule:
        @staticmethod
        def Client():
            return FakeClient(path=None)

        @staticmethod
        def PersistentClient(path):
            return FakeClient(path=path)

    monkeypatch.setattr("turboagents.rag.chroma._chroma_available", lambda: True)
    monkeypatch.setattr(importlib, "import_module", lambda name: FakeChromaModule)

    first = np.zeros((1, 64), dtype=np.float32)
    first[0, 0] = 1.0
    second = np.zeros((1, 64), dtype=np.float32)
    second[0, 1] = 1.0

    index = TurboChroma(path="/tmp/fake-chroma-reset", dim=64, bits=3.5, seed=4)
    index.create_collection("docs", first)
    index.create_collection("docs", second)

    reopened = TurboChroma(path="/tmp/fake-chroma-reset", dim=64, bits=3.5, seed=4)
    reopened.open_collection("docs")
    results = reopened.search(second[0], k=1, rerank_top=None)

    assert len(results) == 1
    assert results[0]["index"] == 0


def test_turbo_lancedb_search_returns_hits() -> None:
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((16, 64), dtype=np.float32)
    query = vectors[3].copy()

    with tempfile.TemporaryDirectory(prefix="turboagents_lancedb_") as tmpdir:
        index = TurboLanceDB(str(Path(tmpdir) / "db"), dim=64, bits=3.5, seed=1)
        index.create_table("embeddings", vectors)
        results = index.search(query, k=3, rerank_top=8)

        assert len(results) == 3
        assert results[0]["index"] == 3


def test_turbo_lancedb_reopened_table_falls_back_without_rerank_state() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((12, 64), dtype=np.float32)
    query = vectors[5].copy()

    with tempfile.TemporaryDirectory(prefix="turboagents_lancedb_reopen_") as tmpdir:
        uri = str(Path(tmpdir) / "db")
        writer = TurboLanceDB(uri, dim=64, bits=3.5, seed=7)
        writer.create_table("embeddings", vectors)

        reopened = TurboLanceDB(uri, dim=64, bits=3.5, seed=7)
        reopened.open_table("embeddings")
        results = reopened.search(query, k=3, rerank_top=8)

        assert len(results) == 3


def test_turbo_lancedb_open_table_retries_after_timeout(monkeypatch) -> None:
    store = TurboLanceDB("/tmp/unused", dim=64, bits=3.5, seed=7)

    class FakeDB:
        def __init__(self) -> None:
            self.calls = 0

        def open_table(self, name: str):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("The read operation timed out")
            return {"name": name}

    fake_db = FakeDB()

    def fake_connect():
        return fake_db

    monkeypatch.setattr(store, "_connect", fake_connect)

    store.open_table("embeddings")

    assert fake_db.calls == 2
    assert store._table == {"name": "embeddings"}


def test_turbo_lancedb_search_retries_after_timeout(monkeypatch) -> None:
    store = TurboLanceDB("/tmp/unused", dim=64, bits=3.5, seed=7)
    query = np.ones(64, dtype=np.float32)
    store.table_name = "embeddings"
    store.vectors = [query.copy()]

    class FakeBuilder:
        def __init__(self, parent) -> None:
            self.parent = parent

        def distance_type(self, metric: str):
            return self

        def limit(self, limit: int):
            return self

        def to_list(self):
            self.parent.calls += 1
            if self.parent.calls == 1:
                raise RuntimeError("read operation timed out")
            return [{"id": 0, "_distance": 1.0, "metadata": {"content": "ok"}}]

    class FakeTable:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query_arr, vector_column_name="vector"):
            return FakeBuilder(self)

    fake_table = FakeTable()
    reopen_calls: list[str] = []

    def fake_open_table(name: str) -> None:
        reopen_calls.append(name)
        store._table = fake_table

    monkeypatch.setattr(store, "open_table", fake_open_table)
    store._table = fake_table

    results = store.search(query, k=1, rerank_top=None)

    assert reopen_calls == ["embeddings"]
    assert results[0]["index"] == 0


def test_turbo_surrealdb_search_returns_hits() -> None:
    rng = np.random.default_rng(4)
    vectors = rng.standard_normal((8, 64), dtype=np.float32)
    query = vectors[2].copy()

    async def scenario() -> list[dict[str, object]]:
        store = TurboSurrealDB(
            url="mem://",
            namespace="testns",
            database="testdb",
            dim=64,
            bits=3.5,
            seed=4,
        )
        await store.create_collection("documents", dim=64)
        await store.add(vectors)
        return await store.search(query, k=3, rerank_top=6)

    results = asyncio.run(scenario())
    assert len(results) == 3
    assert results[0]["index"] == 2


def test_turbo_pgvector_falls_back_without_database() -> None:
    rng = np.random.default_rng(5)
    vectors = rng.standard_normal((10, 64), dtype=np.float32)
    query = vectors[4].copy()

    store = TurboPgvector("postgresql://invalid-host/db", dim=64, bits=3.5, seed=5)
    store.connect = lambda: (_ for _ in ()).throw(RuntimeError("db unavailable"))  # type: ignore[method-assign]
    store.add(vectors)
    results = store.search(query, k=3, rerank_top=6)

    assert len(results) == 3


def test_turbo_pgvector_live_sql_path_with_mock_connection() -> None:
    rng = np.random.default_rng(6)
    vectors = rng.standard_normal((4, 64), dtype=np.float32)
    query = vectors[1].copy()
    payload = quantize(vectors[1], TurboPgvector("postgresql://db", dim=64).config)

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            self.conn.executed.append((sql, params))

        def executemany(self, sql, rows):
            self.conn.executed.append((sql, list(rows)))

        def fetchall(self):
            return [(7, {"title": "doc"}, payload.to_bytes(), 0.8)]

    class FakeConnection:
        def __init__(self):
            self.executed = []
            self.commits = 0

        def cursor(self):
            return FakeCursor(self)

        def commit(self):
            self.commits += 1

        def close(self):
            return None

    conn = FakeConnection()
    store = TurboPgvector("postgresql://db", dim=64, bits=3.5, seed=6)
    store.connect = lambda: conn  # type: ignore[method-assign]
    store.dequantize_payload = lambda compressed: query.copy()  # type: ignore[method-assign]

    text = store.compress_existing()
    store.add(vectors, metadata=[{"i": idx} for idx in range(4)])
    results = store.search(query, k=1, rerank_top=1)

    assert "pgvector adapter ready" in text
    assert conn.commits == 2
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in sql for sql, _ in conn.executed)
    assert any("INSERT INTO documents" in sql for sql, _ in conn.executed if isinstance(sql, str))
    assert results[0]["index"] == 7
