"""Base interfaces for RAG adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from turboagents.quant import Config, inner_product, quantize

@dataclass(slots=True)
class RagAdapterStatus:
    name: str
    experimental: bool = False
    available: bool = False
    detail: str = "not checked"


@dataclass(slots=True)
class SearchHit:
    index: int
    score: float
    metadata: Any = None


@dataclass(slots=True)
class InMemoryTurboIndex:
    dim: int
    bits: float = 3.5
    seed: int = 0
    vectors: list[np.ndarray] = field(default_factory=list)
    compressed: list[Any] = field(default_factory=list)
    metadata: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.config = Config(bits=self.bits, head_dim=self.dim, seed=self.seed, mode="prod")

    def add(self, vectors: np.ndarray, metadata: list[Any] | None = None) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {arr.shape[1]}.")
        meta = metadata or [None] * arr.shape[0]
        if len(meta) != arr.shape[0]:
            raise ValueError("metadata length must match vector count.")
        for vec, item_meta in zip(arr, meta):
            self.vectors.append(vec.copy())
            self.compressed.append(quantize(vec, self.config))
            self.metadata.append(item_meta)

    def search(self, query: np.ndarray, k: int = 10, rerank_top: int | None = None) -> list[dict[str, Any]]:
        query_arr = np.asarray(query, dtype=np.float32)
        if query_arr.shape != (self.dim,):
            raise ValueError(f"Expected query shape {(self.dim,)}, got {query_arr.shape}.")
        scores = np.array(
            [inner_product(query_arr, item, self.config) for item in self.compressed],
            dtype=np.float32,
        )
        top_n = max(k, rerank_top or k)
        candidate_idx = np.argsort(scores)[::-1][:top_n]
        if rerank_top:
            rerank_scores = np.array(
                [float(np.dot(query_arr, self.vectors[idx])) for idx in candidate_idx],
                dtype=np.float32,
            )
            reranked = candidate_idx[np.argsort(rerank_scores)[::-1][:k]]
            final_scores = {
                int(idx): float(np.dot(query_arr, self.vectors[idx])) for idx in reranked
            }
            ordered = reranked
        else:
            ordered = candidate_idx[:k]
            final_scores = {int(idx): float(scores[idx]) for idx in ordered}

        return [
            {
                "index": int(idx),
                "score": final_scores[int(idx)],
                "metadata": self.metadata[int(idx)],
            }
            for idx in ordered
        ]
