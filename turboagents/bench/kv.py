"""Synthetic KV benchmark helpers."""

from __future__ import annotations

import time

import numpy as np

from turboagents.bench.datasets import make_vector_dataset
from turboagents.bench.report import Report
from turboagents.quant import Config, ContextCalculator, dequantize, quantize
from turboagents.quant.config import SUPPORTED_BITS


def build_kv_report() -> Report:
    vectors, queries = make_vector_dataset("tiny-kv")
    query = queries[0]
    calc = ContextCalculator(model="synthetic-8b", memory_gb=24, head_dim=128)

    payload: dict[str, float | int] = {
        "dataset": "tiny-kv",
        "vectors": int(vectors.shape[0]),
        "head_dim": int(vectors.shape[1]),
    }
    for bits in SUPPORTED_BITS:
        cfg = Config(bits=bits, head_dim=128, seed=42, mode="mse")
        start = time.perf_counter()
        compressed = [quantize(vec, cfg) for vec in vectors]
        quant_s = time.perf_counter() - start

        start = time.perf_counter()
        restored = np.stack([dequantize(item, cfg) for item in compressed], axis=0)
        dequant_s = time.perf_counter() - start

        mse = float(np.mean((vectors - restored) ** 2))
        cosine = float(
            np.mean(
                np.sum(vectors * restored, axis=1)
                / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(restored, axis=1) + 1e-8)
            )
        )
        exact_ips = np.dot(vectors, query)
        approx_ips = np.dot(restored, query)
        ip_mae = float(np.mean(np.abs(exact_ips - approx_ips)))
        payload[f"b{bits}_compression_ratio_vs_fp16"] = round(cfg.compression_ratio_vs_fp16, 4)
        payload[f"b{bits}_max_context_24gb"] = calc.max_context(cfg.bits)
        payload[f"b{bits}_quantize_seconds"] = round(quant_s, 6)
        payload[f"b{bits}_dequantize_seconds"] = round(dequant_s, 6)
        payload[f"b{bits}_mean_payload_bytes"] = round(
            float(np.mean([item.estimated_size_bytes for item in compressed])),
            4,
        )
        payload[f"b{bits}_mse"] = round(mse, 8)
        payload[f"b{bits}_mean_cosine_similarity"] = round(cosine, 8)
        payload[f"b{bits}_ip_mae"] = round(ip_mae, 8)

    return Report(
        title="KV Benchmark",
        payload=payload,
    )


def run_kv_benchmark(fmt: str = "text") -> str:
    return build_kv_report().render(fmt)
