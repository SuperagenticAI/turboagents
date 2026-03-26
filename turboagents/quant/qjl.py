"""Initial QJL stage scaffolding with structured residual storage."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_GROUP_SIZE = 16
PROJECTION_SEED_OFFSET = 0x5F3759DF


@lru_cache(maxsize=64)
def _cached_projection_matrix(dim: int, seed: int) -> tuple[tuple[float, ...], ...]:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    return tuple(tuple(float(item) for item in row) for row in matrix)


def projection_matrix(dim: int, seed: int) -> np.ndarray:
    matrix = np.asarray(_cached_projection_matrix(dim, seed + PROJECTION_SEED_OFFSET), dtype=np.float32)
    matrix.setflags(write=False)
    return matrix


def residual_signs(vector: np.ndarray) -> np.ndarray:
    """Return sign bits for a residual-like vector."""
    signs = np.where(vector >= 0.0, 1, -1).astype(np.int8)
    return signs


def encode_residual(
    residual: np.ndarray,
    *,
    seed: int = 0,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Encode a residual using the paper-style QJL sign sketch.

    The paper uses qjl = sign(S r) and reconstructs with
    sqrt(pi / 2) / d * ||r|| * S^T qjl.
    """
    residual_norm = float(np.linalg.norm(residual))
    if residual.size == 0:
        return np.empty(0, dtype=np.int8), 0.0, np.empty(0, dtype=np.float32)
    if residual_norm == 0.0:
        return np.ones(residual.size, dtype=np.int8), 0.0, np.empty(0, dtype=np.float32)
    sketch = projection_matrix(int(residual.size), seed) @ residual.astype(np.float32)
    signs = residual_signs(sketch)
    return signs, residual_norm, np.empty(0, dtype=np.float32)


def decode_residual(
    signs: np.ndarray,
    residual_norm: float,
    group_norms: np.ndarray | None = None,
    *,
    seed: int = 0,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> np.ndarray:
    """Decode a residual estimate from a QJL sign sketch."""
    if signs.size == 0:
        return signs.astype(np.float32)
    if residual_norm == 0.0:
        return np.zeros(signs.size, dtype=np.float32)
    proj = projection_matrix(int(signs.size), seed)
    scale = np.float32(np.sqrt(np.pi / 2.0) / float(signs.size) * residual_norm)
    return scale * (proj.T @ signs.astype(np.float32))


def inner_product(
    query: np.ndarray,
    signs: np.ndarray,
    residual_norm: float,
    group_norms: np.ndarray | None = None,
    *,
    seed: int = 0,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> float:
    """Estimate query · residual directly from the QJL sketch."""
    if query.size != signs.size:
        raise ValueError(
            f"Query length {query.size} does not match residual length {signs.size}."
        )
    if signs.size == 0 or residual_norm == 0.0:
        return 0.0
    proj = projection_matrix(int(signs.size), seed)
    scale = np.float32(np.sqrt(np.pi / 2.0) / float(signs.size) * residual_norm)
    projected_query = proj @ query.astype(np.float32)
    return float(scale * np.dot(projected_query, signs.astype(np.float32)))
