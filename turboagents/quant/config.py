"""Configuration for TurboQuant-style operations."""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_BITS = (2.0, 2.5, 3.0, 3.5, 4.0)
SUPPORTED_HEAD_DIMS = (64, 128, 256)
SUPPORTED_MODES = ("mse", "prod")


@dataclass(frozen=True, slots=True)
class Config:
    """User-facing quantization config."""

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
        if self.head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"Unsupported head_dim={self.head_dim!r}. Expected one of {SUPPORTED_HEAD_DIMS}."
            )
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode={self.mode!r}. Expected one of {SUPPORTED_MODES}."
            )

    @property
    def compression_ratio_vs_fp16(self) -> float:
        """Approximate ratio using FP16 as the baseline."""
        return 16.0 / self.bits

