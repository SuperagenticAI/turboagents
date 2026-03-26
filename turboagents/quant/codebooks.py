"""Codebook helpers for angle quantization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from turboagents.quant.config import SUPPORTED_BITS


@dataclass(frozen=True, slots=True)
class Codebook:
    bits: float
    centroids: np.ndarray
    boundaries: np.ndarray
    angle_type: str
    remaining_dim: int


def level_count(bits: float) -> int:
    """Return an integer level count for a nominal fractional-bit setting."""
    return max(2, int(round(2 ** float(bits))))


def inner_angle_exponent(remaining_dim: int) -> int:
    if remaining_dim < 2:
        raise ValueError("remaining_dim must be at least 2 for an inner polar angle.")
    return max(0, remaining_dim - 2)


def _weighted_quantiles(
    grid: np.ndarray,
    weights: np.ndarray,
    levels: int,
) -> np.ndarray:
    cumulative = np.cumsum(weights, dtype=np.float64)
    cumulative /= cumulative[-1]
    targets = (np.arange(levels, dtype=np.float64) + 0.5) / levels
    return np.interp(targets, cumulative, grid).astype(np.float64)


def _lloyd_max(
    low: float,
    high: float,
    levels: int,
    weights: np.ndarray,
    grid: np.ndarray,
    iterations: int = 64,
) -> np.ndarray:
    centroids = _weighted_quantiles(grid, weights, levels)
    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        buckets = np.searchsorted(boundaries, grid, side="right")
        updated = centroids.copy()
        for idx in range(levels):
            mask = buckets == idx
            bucket_w = weights[mask]
            if bucket_w.size == 0 or float(bucket_w.sum()) == 0.0:
                continue
            bucket_x = grid[mask]
            updated[idx] = float(np.sum(bucket_x * bucket_w) / np.sum(bucket_w))
        updated[0] = max(low, updated[0])
        updated[-1] = min(high, updated[-1])
        if np.allclose(updated, centroids, atol=1e-9):
            centroids = updated
            break
        centroids = updated
    return centroids.astype(np.float32)


def _build_boundaries(centroids: np.ndarray, low: float, high: float) -> np.ndarray:
    mids = (centroids[:-1] + centroids[1:]) / 2.0
    boundaries = np.concatenate(
        [
            np.asarray([low], dtype=np.float32),
            mids.astype(np.float32),
            np.asarray([high], dtype=np.float32),
        ]
    )
    return boundaries


@lru_cache(maxsize=512)
def load_codebook(
    bits: float,
    *,
    angle_type: str = "inner",
    remaining_dim: int | None = None,
    exponent: float | None = None,
) -> Codebook:
    """Return a deterministic generated angle codebook.

    For inner spherical angles, the density is proportional to
    ``sin(theta)^(remaining_dim - 2)`` on ``[0, pi]``. The final angle is
    uniform on ``[-pi, pi]``. We generate Lloyd-Max codebooks directly against
    these densities and cache them by bit-width and effective dimension.
    """
    if float(bits) not in SUPPORTED_BITS:
        raise ValueError(f"Unsupported bits={bits!r}.")
    levels = level_count(bits)

    if angle_type == "inner":
        if remaining_dim is None:
            if exponent is None:
                raise ValueError("remaining_dim or exponent must be provided for inner angles.")
            remaining_dim = int(round(float(exponent))) + 2
        grid = np.linspace(0.0, np.pi, 32768, dtype=np.float64)
        power = inner_angle_exponent(int(remaining_dim))
        weights = np.sin(grid) ** power
        weights += 1e-18
        low, high = 0.0, float(np.pi)
        centroids = _lloyd_max(low, high, levels, weights, grid)
        boundaries = _build_boundaries(centroids, low, high)
        return Codebook(
            bits=float(bits),
            centroids=centroids,
            boundaries=boundaries,
            angle_type=angle_type,
            remaining_dim=int(remaining_dim),
        )

    if angle_type == "last":
        grid = np.linspace(-np.pi, np.pi, 32768, dtype=np.float64)
        weights = np.ones_like(grid, dtype=np.float64)
        low, high = float(-np.pi), float(np.pi)
        centroids = _lloyd_max(low, high, levels, weights, grid)
        boundaries = _build_boundaries(centroids, low, high)
        return Codebook(
            bits=float(bits),
            centroids=centroids,
            boundaries=boundaries,
            angle_type=angle_type,
            remaining_dim=2,
        )

    raise ValueError(f"Unsupported angle_type={angle_type!r}.")


def quantize_value(value: float, codebook: Codebook) -> int:
    """Return the quantization bucket for a scalar value."""
    idx = int(np.searchsorted(codebook.boundaries[1:-1], float(value), side="right"))
    return int(np.clip(idx, 0, codebook.centroids.size - 1))


def dequantize_index(index: int, codebook: Codebook) -> float:
    """Return the centroid value at the provided index."""
    if index < 0 or index >= codebook.centroids.size:
        raise IndexError(f"Codebook index {index} out of range.")
    return float(codebook.centroids[index])
