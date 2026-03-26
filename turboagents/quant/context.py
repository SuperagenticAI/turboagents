"""Context-size estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ContextCalculator:
    """Simple context estimator for KV cache planning.

    This intentionally uses a transparent approximate model until adapter-level
    engine accounting is added.
    """

    model: str
    memory_gb: float
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128

    def bytes_per_token(self, bits: float | str) -> float:
        if bits == "fp16":
            bytes_per_value = 2.0
        else:
            bytes_per_value = float(bits) / 8.0
        return (
            self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * 2
            * bytes_per_value
        )

    def max_context(self, bits: float | str) -> int:
        total_bytes = self.memory_gb * (1024 ** 3)
        return int(total_bytes // self.bytes_per_token(bits))

