"""LanceDB-backed TurboRAG adapter."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from turboagents.rag.base import InMemoryTurboIndex, RagAdapterStatus


def _lancedb_available() -> bool:
    return importlib.util.find_spec("lancedb") is not None


def status() -> RagAdapterStatus:
    if _lancedb_available():
        return RagAdapterStatus(
            name="lancedb",
            available=True,
            detail="upstream LanceDB-backed adapter with TurboQuant rerank",
        )
    return RagAdapterStatus(
        name="lancedb",
        available=True,
        detail="fallback in-memory adapter only; lancedb package not installed",
    )


class TurboLanceDB(InMemoryTurboIndex):
    """Store vectors in LanceDB and rerank candidates with TurboQuant."""

    def __init__(
        self,
        uri: str,
        *,
        dim: int = 1536,
        bits: float = 3.5,
        seed: int = 0,
        vector_column: str = "vector",
        id_column: str = "id",
        metric: str = "dot",
    ) -> None:
        super().__init__(dim=dim, bits=bits, seed=seed)
        self.uri = uri
        self.vector_column = vector_column
        self.id_column = id_column
        self.metric = metric
        self.table_name: str | None = None
        self._db = None
        self._table = None
        self._next_id = 0
        self._rows: list[dict[str, Any]] = []

    def _connect(self):
        if self._db is None:
            if not _lancedb_available():
                raise RuntimeError(
                    "lancedb is not installed. Install with `pip install turboagents[rag]`."
                )
            lancedb = importlib.import_module("lancedb")
            self._db = lancedb.connect(self.uri)
        return self._db

    def create_table(
        self,
        name: str,
        data: np.ndarray,
        metadata: list[object] | None = None,
        *,
        mode: str = "overwrite",
    ) -> None:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {arr.shape[1]}.")
        self.table_name = name
        self.vectors.clear()
        self.compressed.clear()
        self.metadata.clear()
        self._rows.clear()
        self._next_id = 0
        self.add(arr, metadata=metadata)
        db = self._connect()
        self._table = db.create_table(name, data=self._rows, mode=mode)

    def open_table(self, name: str) -> None:
        db = self._connect()
        self.table_name = name
        self._table = db.open_table(name)

    def add(self, vectors: np.ndarray, metadata: list[Any] | None = None) -> None:  # type: ignore[override]
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {arr.shape[1]}.")
        meta = metadata or [None] * arr.shape[0]
        if len(meta) != arr.shape[0]:
            raise ValueError("metadata length must match vector count.")

        rows: list[dict[str, Any]] = []
        for vec, item_meta in zip(arr, meta):
            row_id = self._next_id
            self._next_id += 1
            super().add(vec.reshape(1, -1), metadata=[item_meta])
            row = {
                self.id_column: row_id,
                self.vector_column: vec.tolist(),
                "metadata": item_meta,
            }
            rows.append(row)
            self._rows.append(row)

        if self._table is not None:
            self._table.add(rows)

    def create_native_index(
        self,
        *,
        index_type: str = "IVF_PQ",
        num_partitions: int = 64,
        num_sub_vectors: int = 16,
        num_bits: int = 8,
    ) -> None:
        if self._table is None:
            raise RuntimeError("Table has not been created or opened.")
        self._table.create_index(
            metric=self.metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name=self.vector_column,
            index_type=index_type,
            num_bits=num_bits,
        )

    def search(self, query: np.ndarray, k: int = 10, rerank_top: int | None = None) -> list[dict[str, Any]]:  # type: ignore[override]
        query_arr = np.asarray(query, dtype=np.float32)
        if query_arr.shape != (self.dim,):
            raise ValueError(f"Expected query shape {(self.dim,)}, got {query_arr.shape}.")

        if self._table is None:
            return super().search(query_arr, k=k, rerank_top=rerank_top)

        candidate_limit = max(k, rerank_top or k)
        builder = self._table.search(query_arr, vector_column_name=self.vector_column)
        if hasattr(builder, "distance_type"):
            builder = builder.distance_type(self.metric)
        records = builder.limit(candidate_limit).to_list()

        if not records:
            return []

        if rerank_top:
            ranked = []
            for record in records:
                idx = int(record[self.id_column])
                score = float(np.dot(query_arr, self.vectors[idx]))
                ranked.append((score, idx, record.get("metadata")))
            ranked.sort(key=lambda item: item[0], reverse=True)
            ranked = ranked[:k]
            return [
                {"index": idx, "score": score, "metadata": metadata}
                for score, idx, metadata in ranked
            ]

        return [
            {
                "index": int(record[self.id_column]),
                "score": float(-record.get("_distance", 0.0) if self.metric == "l2" else record.get("_distance", 0.0)),
                "metadata": record.get("metadata"),
            }
            for record in records[:k]
        ]
