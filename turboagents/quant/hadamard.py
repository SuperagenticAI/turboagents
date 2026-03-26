"""Fast Walsh-Hadamard Transform utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

from turboagents.quant.config import Config


def _as_float_array(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("Vector must not be empty.")
    if arr.size & (arr.size - 1):
        raise ValueError(
            f"FWHT requires power-of-two length, got length {arr.size}."
        )
    return arr.copy()


@lru_cache(maxsize=128)
def _cached_sign_pattern(length: int, seed: int) -> tuple[float, ...]:
    rng = np.random.default_rng(seed)
    values = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=length)
    return tuple(float(item) for item in values)


def sign_pattern(length: int, seed: int) -> np.ndarray:
    """Generate a deterministic Rademacher sign pattern."""
    if length <= 0:
        raise ValueError("length must be positive.")
    arr = np.asarray(_cached_sign_pattern(length, seed), dtype=np.float32)
    arr.setflags(write=False)
    return arr


def fwht(vector: Iterable[float], normalize: bool = True) -> np.ndarray:
    """Compute the in-place Fast Walsh-Hadamard Transform."""
    arr = _as_float_array(vector)
    h = 1
    while h < arr.size:
        step = h * 2
        for start in range(0, arr.size, step):
            left = arr[start : start + h].copy()
            right = arr[start + h : start + step].copy()
            arr[start : start + h] = left + right
            arr[start + h : start + step] = left - right
        h = step
    if normalize:
        arr /= np.float32(np.sqrt(float(arr.size)))
    return arr


def rotate(vector: Iterable[float], config: Config) -> np.ndarray:
    """Apply seeded sign flips followed by a normalized FWHT."""
    arr = _as_float_array(vector)
    if arr.size != config.head_dim:
        raise ValueError(
            f"Vector length {arr.size} does not match config.head_dim={config.head_dim}."
        )
    return fwht(arr * sign_pattern(arr.size, config.seed), normalize=True)


def inverse_rotate(vector: Iterable[float], config: Config) -> np.ndarray:
    """Invert the seeded rotation.

    The normalized Hadamard transform is self-inverse, so inversion is another
    Hadamard transform followed by the same sign pattern.
    """
    arr = _as_float_array(vector)
    if arr.size != config.head_dim:
        raise ValueError(
            f"Vector length {arr.size} does not match config.head_dim={config.head_dim}."
        )
    return fwht(arr, normalize=True) * sign_pattern(arr.size, config.seed)
