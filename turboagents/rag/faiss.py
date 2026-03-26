"""FAISS-backed TurboRAG adapter."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from turboagents.rag.base import InMemoryTurboIndex, RagAdapterStatus


def _faiss_available() -> bool:
    return importlib.util.find_spec("faiss") is not None


def status() -> RagAdapterStatus:
    if _faiss_available():
        return RagAdapterStatus(
            name="faiss",
            available=True,
            detail="real FAISS adapter available",
        )
    return RagAdapterStatus(
        name="faiss",
        available=False,
        detail="faiss package not installed",
    )


class TurboFAISS(InMemoryTurboIndex):
    """TurboRAG adapter backed by a FAISS inner-product index."""

    def __init__(
        self,
        dim: int = 1536,
        *,
        bits: float = 3.5,
        seed: int = 0,
        metric: str = "ip",
    ) -> None:
        super().__init__(dim=dim, bits=bits, seed=seed)
        self.metric = metric
        self._faiss = None
        self._index = None
        self._ids: list[int] = []

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        if not _faiss_available():
            raise RuntimeError("faiss is not installed. Install turboagents[rag].")
        import faiss

        if self.metric != "ip":
            raise ValueError(f"Unsupported FAISS metric={self.metric!r}. Only 'ip' is supported.")
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self.dim)

    def add(self, vectors: np.ndarray, metadata: list[Any] | None = None) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._ensure_index()
        start = len(self.vectors)
        super().add(arr, metadata=metadata)
        self._index.add(arr)
        self._ids.extend(range(start, start + arr.shape[0]))

    def search(self, query: np.ndarray, k: int = 10, rerank_top: int | None = None) -> list[dict[str, Any]]:
        query_arr = np.asarray(query, dtype=np.float32)
        if query_arr.shape != (self.dim,):
            raise ValueError(f"Expected query shape {(self.dim,)}, got {query_arr.shape}.")
        self._ensure_index()
        if self._index is None or self._index.ntotal == 0:
            return []

        top_n = max(k, rerank_top or k)
        scores, indices = self._index.search(query_arr.reshape(1, -1), top_n)
        candidate_idx = [int(idx) for idx in indices[0].tolist() if idx >= 0]
        if not candidate_idx:
            return []

        if rerank_top:
            rerank_scores = np.array(
                [float(np.dot(query_arr, self.vectors[idx])) for idx in candidate_idx],
                dtype=np.float32,
            )
            order = np.argsort(rerank_scores)[::-1][:k]
            final_idx = [candidate_idx[int(pos)] for pos in order]
            final_scores = {idx: float(np.dot(query_arr, self.vectors[idx])) for idx in final_idx}
        else:
            final_idx = candidate_idx[:k]
            final_scores = {
                idx: float(scores[0][pos])
                for pos, idx in enumerate(candidate_idx[:k])
            }

        return [
            {
                "index": idx,
                "score": final_scores[idx],
                "metadata": self.metadata[idx],
            }
            for idx in final_idx
        ]
