"""Polar-coordinate helpers for the first structured PolarQuant stage."""

from __future__ import annotations

import numpy as np

from turboagents.quant.codebooks import dequantize_index, load_codebook, quantize_value
from turboagents.quant.config import Config
from turboagents.quant.hadamard import rotate


def _safe_arccos(value: float) -> float:
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def to_spherical(vector: np.ndarray) -> tuple[float, np.ndarray]:
    """Convert a Cartesian vector to radius and spherical angles."""
    radius = float(np.linalg.norm(vector))
    if radius == 0.0:
        return 0.0, np.zeros(vector.size - 1, dtype=np.float32)

    angles = np.zeros(vector.size - 1, dtype=np.float32)
    for idx in range(vector.size - 2):
        tail_norm = float(np.linalg.norm(vector[idx:]))
        if tail_norm == 0.0:
            angles[idx] = 0.0
        else:
            angles[idx] = _safe_arccos(float(vector[idx]) / tail_norm)
    angles[-1] = float(np.arctan2(vector[-1], vector[-2]))
    return radius, angles


def from_spherical(radius: float, angles: np.ndarray) -> np.ndarray:
    """Convert radius and spherical angles back to Cartesian coordinates."""
    dim = angles.size + 1
    vector = np.zeros(dim, dtype=np.float32)
    if radius == 0.0:
        return vector

    if dim == 1:
        vector[0] = radius
        return vector

    vector[0] = radius * np.cos(float(angles[0]))

    sin_product = 1.0
    for idx in range(1, dim - 1):
        sin_product *= np.sin(float(angles[idx - 1]))
        vector[idx] = radius * sin_product * np.cos(float(angles[idx]))

    if dim > 2:
        vector[-1] = radius * np.prod(np.sin(angles[:-1]), dtype=np.float32) * np.sin(
            float(angles[-1])
        )
    else:
        vector[-1] = radius * np.sin(float(angles[-1]))
    return vector


def polar_quantize_rotated(rotated: np.ndarray, config: Config) -> tuple[np.ndarray, float]:
    """Quantize a vector that is already in the rotated domain."""
    radius, angles = to_spherical(rotated)
    indices = np.zeros_like(angles, dtype=np.int32)
    for idx, angle in enumerate(angles):
        angle_type = "last" if idx == angles.size - 1 else "inner"
        remaining_dim = config.head_dim - idx
        codebook = load_codebook(config.bits, angle_type=angle_type, remaining_dim=remaining_dim)
        indices[idx] = quantize_value(float(angle), codebook)
    return indices, radius


def polar_quantize(vector: np.ndarray, config: Config) -> tuple[np.ndarray, float]:
    """Rotate a vector and quantize its spherical angles."""
    return polar_quantize_rotated(rotate(vector, config), config)


def polar_dequantize(angle_indices: np.ndarray, radius: float, config: Config) -> np.ndarray:
    """Reconstruct the rotated vector from quantized spherical angles."""
    angles = np.zeros(angle_indices.size, dtype=np.float32)
    for idx, angle_index in enumerate(angle_indices):
        angle_type = "last" if idx == angle_indices.size - 1 else "inner"
        remaining_dim = config.head_dim - idx
        codebook = load_codebook(config.bits, angle_type=angle_type, remaining_dim=remaining_dim)
        angles[idx] = dequantize_index(int(angle_index), codebook)
    return from_spherical(radius, angles)
