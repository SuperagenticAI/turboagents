"""Chroma-backed TurboRAG adapter."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import numpy as np

from turboagents.rag.base import InMemoryTurboIndex, RagAdapterStatus


def _chroma_available() -> bool:
    return importlib.util.find_spec("chromadb") is not None


def status() -> RagAdapterStatus:
    if _chroma_available():
        return RagAdapterStatus(
            name="chroma",
            available=True,
            detail="Chroma-backed adapter with TurboQuant rerank",
        )
    return RagAdapterStatus(
        name="chroma",
        available=False,
        detail="chromadb package not installed",
    )


class TurboChroma(InMemoryTurboIndex):
    """Store vectors in Chroma and rerank candidates with TurboQuant."""

    def __init__(
        self,
        path: str | None = None,
        *,
        collection_name: str | None = None,
        dim: int = 1536,
        bits: float = 3.5,
        seed: int = 0,
        metric: str = "cosine",
    ) -> None:
        super().__init__(dim=dim, bits=bits, seed=seed)
        self.path = path
        self.collection_name = collection_name
        self.metric = metric.lower()
        self._client = None
        self._collection = None
        self._next_id = 0

    @staticmethod
    def _collection_size(collection: Any) -> int:
        count = getattr(collection, "count", None)
        if callable(count):
            try:
                return int(count())
            except Exception:
                return 0
        return 0

    def _connect(self):
        if self._client is None:
            if not _chroma_available():
                raise RuntimeError(
                    "chromadb is not installed. Install with `uv sync --extra rag`."
                )
            chromadb = importlib.import_module("chromadb")
            if self.path:
                self._client = chromadb.PersistentClient(path=self.path)
            else:
                self._client = chromadb.Client()
        return self._client

    @staticmethod
    def _normalize_metadata(value: Any, *, row_id: int) -> dict[str, Any]:
        if value is None:
            return {"doc_id": str(row_id)}
        if not isinstance(value, dict):
            return {"doc_id": str(row_id), "value": str(value)}

        normalized: dict[str, Any] = {"doc_id": str(row_id)}
        for key, item in value.items():
            if item is None:
                continue
            if isinstance(item, (str, int, float, bool)):
                normalized[str(key)] = item
            else:
                normalized[str(key)] = str(item)
        return normalized

    def create_collection(
        self,
        name: str,
        data: np.ndarray | None = None,
        metadata: list[Any] | None = None,
    ) -> None:
        client = self._connect()
        self.collection_name = name
        delete_collection = getattr(client, "delete_collection", None)
        if callable(delete_collection):
            try:
                delete_collection(name=name)
            except Exception:
                pass
        self._collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": self.metric},
        )
        self.vectors.clear()
        self.compressed.clear()
        self.metadata.clear()
        self._next_id = 0
        if data is not None:
            self.add(data, metadata=metadata)

    def open_collection(self, name: str) -> None:
        client = self._connect()
        self.collection_name = name
        self._collection = client.get_collection(name=name)
        self._next_id = self._collection_size(self._collection)

    def add(self, vectors: np.ndarray, metadata: list[Any] | None = None) -> None:  # type: ignore[override]
        if self._collection is None:
            name = self.collection_name or "documents"
            self.create_collection(name)

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {arr.shape[1]}.")
        meta = metadata or [None] * arr.shape[0]
        if len(meta) != arr.shape[0]:
            raise ValueError("metadata length must match vector count.")

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        for vec, item_meta in zip(arr, meta):
            row_id = self._next_id
            self._next_id += 1
            super().add(vec.reshape(1, -1), metadata=[item_meta])
            ids.append(str(row_id))
            embeddings.append(vec.tolist())
            payload = self._normalize_metadata(item_meta, row_id=row_id)
            metadatas.append(payload)

        self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(  # type: ignore[override]
        self, query: np.ndarray, k: int = 10, rerank_top: int | None = None
    ) -> list[dict[str, Any]]:
        query_arr = np.asarray(query, dtype=np.float32)
        if query_arr.shape != (self.dim,):
            raise ValueError(f"Expected query shape {(self.dim,)}, got {query_arr.shape}.")

        if self._collection is None:
            return super().search(query_arr, k=k, rerank_top=rerank_top)

        candidate_limit = max(k, rerank_top or k)
        response = self._collection.query(
            query_embeddings=[query_arr.tolist()],
            n_results=candidate_limit,
            include=["metadatas", "distances"],
        )
        raw_ids = (response.get("ids") or [[]])[0]
        if not raw_ids:
            return []
        raw_meta = (response.get("metadatas") or [[]])[0]
        raw_distances = (response.get("distances") or [[]])[0]

        records: list[tuple[int, Any, float]] = []
        for idx, raw_id in enumerate(raw_ids):
            records.append(
                (
                    int(raw_id),
                    raw_meta[idx] if idx < len(raw_meta) else None,
                    float(raw_distances[idx]) if idx < len(raw_distances) else 0.0,
                )
            )

        if rerank_top:
            can_rerank = True
            for index, _, _ in records:
                if index >= len(self.vectors):
                    can_rerank = False
                    break
            if not can_rerank:
                rerank_top = None

        if rerank_top:
            ranked = []
            for index, metadata, _distance in records:
                score = float(np.dot(query_arr, self.vectors[index]))
                ranked.append((score, index, metadata))
            ranked.sort(key=lambda item: item[0], reverse=True)
            ranked = ranked[:k]
            return [
                {"index": index, "score": score, "metadata": metadata}
                for score, index, metadata in ranked
            ]

        return [
            {
                "index": index,
                "score": float(-distance),
                "metadata": metadata,
            }
            for index, metadata, distance in records[:k]
        ]
