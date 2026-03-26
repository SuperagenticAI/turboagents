"""SurrealDB-backed TurboRAG adapter."""

from __future__ import annotations

import base64
import importlib
import importlib.util
from typing import Any

import numpy as np

from turboagents.quant import Config, inner_product, quantize
from turboagents.quant.types import CompressedVector
from turboagents.rag.base import InMemoryTurboIndex, RagAdapterStatus


def _surrealdb_available() -> bool:
    return importlib.util.find_spec("surrealdb") is not None


def status() -> RagAdapterStatus:
    if _surrealdb_available():
        return RagAdapterStatus(
            name="surrealdb",
            experimental=True,
            available=True,
            detail="async upstream client adapter with native HNSW candidate search",
        )
    return RagAdapterStatus(
        name="surrealdb",
        experimental=True,
        available=True,
        detail="fallback in-memory adapter only; surrealdb package not installed",
    )


class TurboSurrealDB(InMemoryTurboIndex):
    """Use SurrealDB for storage/candidate search and TurboQuant for rerank."""

    def __init__(
        self,
        *,
        url: str,
        namespace: str,
        database: str,
        dim: int = 1536,
        bits: float = 3.5,
        seed: int = 0,
        metric: str = "COSINE",
        auth: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(dim=dim, bits=bits, seed=seed)
        self.url = url
        self.namespace = namespace
        self.database = database
        self.metric = metric.upper()
        self.auth = auth
        self.collection: str | None = None
        self._client = None
        self._next_id = 0

    async def _ensure_client(self):
        if self._client is None:
            if not _surrealdb_available():
                raise RuntimeError(
                    "surrealdb is not installed. Install with `uv sync --extra rag`."
                )
            surrealdb = importlib.import_module("surrealdb")
            client = surrealdb.AsyncSurreal(self.url)
            await client.connect()
            if self.auth:
                await client.signin(self.auth)
            await client.use(self.namespace, self.database)
            self._client = client
        return self._client

    @staticmethod
    def _encode_payload(payload: CompressedVector) -> str:
        return base64.b64encode(payload.to_bytes()).decode("ascii")

    @staticmethod
    def _decode_payload(payload: str) -> CompressedVector:
        return CompressedVector.from_bytes(base64.b64decode(payload.encode("ascii")))

    async def create_collection(self, name: str, dim: int | None = None) -> None:
        if dim is not None and dim != self.dim:
            raise ValueError(f"Collection dim {dim} does not match adapter dim {self.dim}.")
        self.collection = name
        client = await self._ensure_client()
        await client.query(f"DEFINE TABLE {name} SCHEMALESS;")
        await client.query(
            f"DEFINE INDEX {name}_embedding_idx ON {name} "
            f"FIELDS embedding HNSW DIMENSION {self.dim} DIST {self.metric} TYPE F32 EFC 150 M 8;"
        )

    async def add(self, embeddings: np.ndarray, metadata: list[Any] | None = None) -> None:  # type: ignore[override]
        if self.collection is None:
            raise RuntimeError("Collection has not been created.")
        client = await self._ensure_client()
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {arr.shape[1]}.")
        meta = metadata or [None] * arr.shape[0]
        if len(meta) != arr.shape[0]:
            raise ValueError("metadata length must match vector count.")

        for vec, item_meta in zip(arr, meta):
            row_id = self._next_id
            self._next_id += 1
            super().add(vec.reshape(1, -1), metadata=[item_meta])
            payload = self._encode_payload(self.compressed[row_id])
            await client.create(
                f"{self.collection}:{row_id}",
                {
                    "embedding": vec.tolist(),
                    "metadata": item_meta,
                    "turbo_payload": payload,
                },
            )

    async def search(
        self,
        query_vec: np.ndarray,
        *,
        k: int = 10,
        rerank_top: int | None = None,
    ) -> list[dict[str, object]]:
        if self.collection is None:
            raise RuntimeError("Collection has not been created.")
        client = await self._ensure_client()
        query_arr = np.asarray(query_vec, dtype=np.float32)
        if query_arr.shape != (self.dim,):
            raise ValueError(f"Expected query shape {(self.dim,)}, got {query_arr.shape}.")

        candidate_limit = max(k, rerank_top or k)
        sql = (
            f"SELECT id, metadata, turbo_payload, vector::distance::knn() AS dist "
            f"FROM {self.collection} "
            f"WHERE embedding <|{candidate_limit},{self.metric}|> $query;"
        )
        rows = await client.query(sql, {"query": query_arr.tolist()})
        if not rows:
            return []

        if rerank_top:
            ranked = []
            for row in rows:
                record_id = str(row["id"]).split(":")[-1]
                idx = int(record_id)
                payload = self._decode_payload(row["turbo_payload"])
                score = inner_product(query_arr, payload, self.config)
                ranked.append((float(score), idx, row.get("metadata")))
            ranked.sort(key=lambda item: item[0], reverse=True)
            ranked = ranked[:k]
            return [
                {"index": idx, "score": score, "metadata": metadata}
                for score, idx, metadata in ranked
            ]

        return [
            {
                "index": int(str(row["id"]).split(":")[-1]),
                "score": float(-row.get("dist", 0.0) if self.metric == "EUCLIDEAN" else row.get("dist", 0.0)),
                "metadata": row.get("metadata"),
            }
            for row in rows[:k]
        ]
