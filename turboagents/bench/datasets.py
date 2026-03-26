"""Small built-in synthetic benchmark datasets.

These helpers deliberately stay lightweight so they are usable on constrained
developer machines. They provide deterministic vector/query batches that let
the CLI, examples, and tests exercise the benchmark surfaces without pulling
large external corpora.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    seed: int
    num_vectors: int
    num_queries: int
    dim: int
    noise_scale: float = 0.02


DATASETS: dict[str, DatasetSpec] = {
    "tiny-kv": DatasetSpec(name="tiny-kv", seed=42, num_vectors=64, num_queries=8, dim=128),
    "tiny-rag": DatasetSpec(name="tiny-rag", seed=7, num_vectors=256, num_queries=16, dim=128),
    "medium-rag": DatasetSpec(name="medium-rag", seed=17, num_vectors=1024, num_queries=32, dim=128),
    "paper-sim": DatasetSpec(name="paper-sim", seed=123, num_vectors=64, num_queries=8, dim=128),
}


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark dataset: {name!r}") from exc


def make_vector_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = get_dataset(name)
    rng = np.random.default_rng(spec.seed)
    base = rng.standard_normal((spec.num_vectors, spec.dim), dtype=np.float32)
    queries = []
    for idx in range(spec.num_queries):
        anchor = base[idx % spec.num_vectors]
        noise = spec.noise_scale * rng.standard_normal(spec.dim, dtype=np.float32)
        queries.append(anchor + noise)
    return base, np.stack(queries, axis=0)
