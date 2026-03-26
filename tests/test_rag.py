import asyncio
from pathlib import Path
import tempfile

import numpy as np

from turboagents.quant import quantize
from turboagents.rag import TurboFAISS, TurboLanceDB, TurboPgvector, TurboSurrealDB


def test_turbo_faiss_search_returns_hits() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((32, 64), dtype=np.float32)
    query = vectors[0].copy()

    index = TurboFAISS(dim=64, bits=3.5, seed=0)
    index.add(vectors)
    results = index.search(query, k=5, rerank_top=10)

    assert len(results) == 5
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
