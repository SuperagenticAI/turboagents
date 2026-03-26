"""pgvector-backed TurboRAG adapter."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from turboagents.quant.types import CompressedVector
from turboagents.rag.base import InMemoryTurboIndex, RagAdapterStatus


def _pgvector_available() -> bool:
    return importlib.util.find_spec("pgvector.psycopg2") is not None


def _psycopg2_available() -> bool:
    return importlib.util.find_spec("psycopg2") is not None


def status() -> RagAdapterStatus:
    if _pgvector_available() and _psycopg2_available():
        return RagAdapterStatus(
            name="pgvector",
            available=True,
            detail="real psycopg2/pgvector adapter available",
        )
    return RagAdapterStatus(
        name="pgvector",
        available=False,
        detail="psycopg2 or pgvector package not installed",
    )


class TurboPgvector(InMemoryTurboIndex):
    def __init__(
        self,
        dsn: str,
        *,
        table: str = "documents",
        id_column: str = "id",
        embedding_column: str = "embedding",
        metadata_column: str = "metadata",
        payload_column: str = "turbo_payload",
        dim: int = 1536,
        bits: float = 3.5,
        seed: int = 0,
    ) -> None:
        super().__init__(dim=dim, bits=bits, seed=seed)
        self.dsn = dsn
        self.table = table
        self.id_column = id_column
        self.embedding_column = embedding_column
        self.metadata_column = metadata_column
        self.payload_column = payload_column
        self._conn = None

    def connect(self):
        if self._conn is not None:
            return self._conn
        if not (_pgvector_available() and _psycopg2_available()):
            raise RuntimeError("psycopg2 and pgvector are required. Install turboagents[rag].")
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(self.dsn)
        register_vector(conn)
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def ensure_schema(self) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    {self.id_column} BIGSERIAL PRIMARY KEY,
                    {self.embedding_column} vector({self.dim}) NOT NULL,
                    {self.metadata_column} JSONB,
                    {self.payload_column} BYTEA NOT NULL
                )
                """
            )
        conn.commit()

    def compress_existing(self) -> str:
        self.ensure_schema()
        return f"pgvector adapter ready for table={self.table} at {self.dsn}"

    def add(self, vectors: np.ndarray, metadata: list[Any] | None = None) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        meta = metadata or [None] * arr.shape[0]
        super().add(arr, metadata=meta)
        try:
            conn = self.connect()
        except Exception:
            return
        rows = []
        start = len(self.compressed) - arr.shape[0]
        for offset, (vec, item_meta) in enumerate(zip(arr, meta)):
            payload = self.compressed[start + offset]
            rows.append((vec.tolist(), item_meta, payload.to_bytes()))
        with conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table} ({self.embedding_column}, {self.metadata_column}, {self.payload_column})
                VALUES (%s, %s, %s)
                """,
                rows,
            )
        conn.commit()

    def add_embeddings(self, embeddings: np.ndarray, metadata: list[object] | None = None) -> None:
        self.add(embeddings, metadata=metadata)

    def search(self, query: np.ndarray, k: int = 10, rerank_top: int | None = None) -> list[dict[str, Any]]:
        query_arr = np.asarray(query, dtype=np.float32)
        if query_arr.shape != (self.dim,):
            raise ValueError(f"Expected query shape {(self.dim,)}, got {query_arr.shape}.")

        try:
            conn = self.connect()
        except Exception:
            return super().search(query_arr, k=k, rerank_top=rerank_top)

        candidate_limit = max(k, rerank_top or k)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    {self.id_column},
                    {self.metadata_column},
                    {self.payload_column},
                    -({self.embedding_column} <#> %s::vector) AS score
                FROM {self.table}
                ORDER BY {self.embedding_column} <#> %s::vector
                LIMIT %s
                """,
                (query_arr.tolist(), query_arr.tolist(), candidate_limit),
            )
            rows = cur.fetchall()

        if not rows:
            return []

        if rerank_top:
            rescored: list[tuple[int, float, Any]] = []
            for row_id, item_meta, payload_bytes, _score in rows:
                payload = CompressedVector.from_bytes(bytes(payload_bytes))
                score = float(np.dot(query_arr, self.dequantize_payload(payload)))
                rescored.append((int(row_id), score, item_meta))
            rescored.sort(key=lambda item: item[1], reverse=True)
            return [
                {"index": row_id, "score": score, "metadata": item_meta}
                for row_id, score, item_meta in rescored[:k]
            ]

        return [
            {
                "index": int(row_id),
                "score": float(score),
                "metadata": item_meta,
            }
            for row_id, item_meta, _payload_bytes, score in rows[:k]
        ]

    def dequantize_payload(self, payload: CompressedVector) -> np.ndarray:
        from turboagents.quant import dequantize

        return dequantize(payload, self.config)
