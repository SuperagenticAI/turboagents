"""Offline compression helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from turboagents.quant import Config, quantize


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    bits: float = 3.5,
    head_dim: int = 128,
    seed: int = 0,
) -> str:
    if not input_path or not output_path:
        return "Compression CLI currently supports --input <npy> --output <npz>."

    source = Path(input_path)
    target = Path(output_path)
    vectors = np.load(source)
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    dim = arr.shape[1]
    cfg = Config(bits=bits, head_dim=head_dim or dim, seed=seed, mode="prod")
    if cfg.head_dim != dim:
        raise ValueError(f"Input vectors have dim={dim}, but head_dim={cfg.head_dim}.")

    payload = [quantize(vec, cfg).to_bytes() for vec in arr]
    np.savez_compressed(
        target,
        payload=np.array(payload, dtype=object),
        bits=np.float32(bits),
        head_dim=np.int32(dim),
        seed=np.int64(seed),
    )
    return f"Compressed {arr.shape[0]} vectors of dim {dim} to {target}"
