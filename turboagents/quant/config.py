"""Configuration for TurboQuant-style operations."""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_BITS = (2.0, 2.5, 3.0, 3.5, 4.0)
SUPPORTED_MODES = ("mse", "prod")


def next_power_of_two(n: int) -> int:
    """Smallest power of two greater than or equal to ``n`` (``n >= 1``)."""
    if n < 1:
        raise ValueError(f"Expected a positive dimension, got {n!r}.")
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True, slots=True)
class Config:
    """User-facing quantization config.

    ``head_dim`` is the logical embedding/head dimension and may be any
    positive integer. The Walsh-Hadamard rotation runs on a zero-padded copy of
    length ``transform_dim`` (the next power of two), so dimensions such as 384,
    768 or 1536 work without the caller padding or truncating embeddings.
    """

    bits: float = 3.5
    head_dim: int = 128
    seed: int = 0
    mode: str = "mse"

    def __post_init__(self) -> None:
        bits = float(self.bits)
        if bits not in SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported bits={self.bits!r}. Expected one of {SUPPORTED_BITS}."
            )
        if not isinstance(self.head_dim, int) or self.head_dim < 1:
            raise ValueError(
                f"Unsupported head_dim={self.head_dim!r}. Expected a positive integer."
            )
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode={self.mode!r}. Expected one of {SUPPORTED_MODES}."
            )

    @property
    def transform_dim(self) -> int:
        """Power-of-two length used for the rotation and quantization stages."""
        return next_power_of_two(self.head_dim)

    @property
    def compression_ratio_vs_fp16(self) -> float:
        """Approximate ratio using FP16 as the baseline."""
        return 16.0 / self.bits

