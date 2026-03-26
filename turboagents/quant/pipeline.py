"""Initial end-to-end quantization pipeline."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from turboagents.quant.config import Config
from turboagents.quant.hadamard import inverse_rotate, rotate
from turboagents.quant.polar import polar_dequantize, polar_quantize_rotated
from turboagents.quant.qjl import decode_residual, encode_residual, inner_product as qjl_inner_product
from turboagents.quant.types import CompressedVector


def _as_vector(vector: Iterable[float], config: Config) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {arr.shape}.")
    if arr.size != config.head_dim:
        raise ValueError(
            f"Vector length {arr.size} does not match config.head_dim={config.head_dim}."
        )
    return arr


def quantize(vector: Iterable[float], config: Config) -> CompressedVector:
    """Structured two-stage quantization in the rotated domain."""
    arr = _as_vector(vector, config)
    rotated = rotate(arr, config)
    angle_indices, radius = polar_quantize_rotated(rotated, config)
    rotated_approx = polar_dequantize(angle_indices, radius, config)
    residual_sign_bits, residual_norm, residual_group_norms = encode_residual(
        rotated - rotated_approx,
        seed=config.seed,
    )
    return CompressedVector(
        angle_indices=angle_indices,
        residual_signs=residual_sign_bits,
        residual_group_norms=residual_group_norms,
        radius=radius,
        residual_norm=residual_norm,
        bits=config.bits,
        seed=config.seed,
        mode=config.mode,
    )


def rotated_estimate(compressed: CompressedVector, config: Config) -> np.ndarray:
    """Return the reconstructed vector in the rotated domain."""
    if compressed.seed != config.seed:
        raise ValueError(
            "Compressed payload seed does not match the provided config seed."
        )
    rotated = polar_dequantize(compressed.angle_indices, compressed.radius, config)
    residual = decode_residual(
        compressed.residual_signs,
        compressed.residual_norm,
        compressed.residual_group_norms,
        seed=config.seed,
    )
    return rotated + residual


def dequantize(compressed: CompressedVector, config: Config) -> np.ndarray:
    """Invert the structured compressed representation."""
    return inverse_rotate(rotated_estimate(compressed, config), config)


def inner_product(query: Iterable[float], compressed: CompressedVector, config: Config) -> float:
    """Compute an inner product in the compressed domain."""
    query_arr = _as_vector(query, config)
    query_rot = rotate(query_arr, config)
    polar_component = polar_dequantize(compressed.angle_indices, compressed.radius, config)
    residual_component = qjl_inner_product(
        query_rot,
        compressed.residual_signs,
        compressed.residual_norm,
        compressed.residual_group_norms,
        seed=config.seed,
    )
    return float(np.dot(query_rot, polar_component) + residual_component)
