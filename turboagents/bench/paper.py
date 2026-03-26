"""Paper-style synthetic evaluation helpers."""

from __future__ import annotations

import numpy as np

from turboagents.bench.datasets import make_vector_dataset
from turboagents.bench.report import Report
from turboagents.quant import Config, dequantize, quantize
from turboagents.quant.config import SUPPORTED_BITS


def build_paper_report() -> Report:
    vectors, _queries = make_vector_dataset("paper-sim")
    payload: dict[str, float | str] = {"dataset": "paper-sim"}
    previous_mse: float | None = None
    for bits in SUPPORTED_BITS:
        cfg = Config(bits=bits, head_dim=128, seed=123, mode="mse")
        restored = np.stack([dequantize(quantize(vec, cfg), cfg) for vec in vectors], axis=0)
        mse = float(np.mean((vectors - restored) ** 2))
        cosine = float(
            np.mean(
                np.sum(vectors * restored, axis=1)
                / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(restored, axis=1) + 1e-8)
            )
        )
        payload[f"mse_b{bits}"] = round(mse, 8)
        payload[f"cosine_b{bits}"] = round(cosine, 8)
        if previous_mse is not None:
            payload[f"mse_delta_from_prev_b{bits}"] = round(mse - previous_mse, 8)
        previous_mse = mse
    return Report(title="Paper Benchmark", payload=payload)


def run_paper_benchmark(fmt: str = "text") -> str:
    return build_paper_report().render(fmt)
