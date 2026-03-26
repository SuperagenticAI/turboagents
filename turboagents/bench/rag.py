"""Synthetic RAG benchmark helpers."""

from __future__ import annotations

import time

import numpy as np

from turboagents.bench.datasets import make_vector_dataset
from turboagents.bench.report import Report
from turboagents.rag.faiss import TurboFAISS
from turboagents.quant.config import SUPPORTED_BITS


def _recall_at_k(pred: list[int], truth: list[int]) -> float:
    truth_set = set(truth)
    return float(sum(1 for item in pred if item in truth_set) / max(1, len(truth)))


def build_rag_report() -> Report:
    base, queries = make_vector_dataset("tiny-rag")
    dim = int(base.shape[1])
    payload: dict[str, float | int] = {
        "dataset": "tiny-rag",
        "dataset_size": int(base.shape[0]),
        "dim": dim,
        "queries": int(queries.shape[0]),
    }

    for bits in SUPPORTED_BITS:
        start = time.perf_counter()
        index = TurboFAISS(dim=dim, bits=bits, seed=7)
        index.add(base)
        build_s = time.perf_counter() - start

        start = time.perf_counter()
        recalls_1: list[float] = []
        recalls_10: list[float] = []
        for query in queries:
            pred = index.search(query, k=10, rerank_top=50)
            exact_scores = np.dot(base, query)
            truth1 = list(np.argsort(exact_scores)[::-1][:1])
            truth10 = list(np.argsort(exact_scores)[::-1][:10])
            recalls_1.append(_recall_at_k([item["index"] for item in pred[:1]], truth1))
            recalls_10.append(_recall_at_k([item["index"] for item in pred], truth10))
        query_s = time.perf_counter() - start

        payload[f"b{bits}_build_seconds"] = round(build_s, 6)
        payload[f"b{bits}_query_seconds"] = round(query_s, 6)
        payload[f"b{bits}_recall_at_1"] = round(float(np.mean(recalls_1)), 6)
        payload[f"b{bits}_recall_at_10"] = round(float(np.mean(recalls_10)), 6)

    return Report(
        title="RAG Benchmark",
        payload=payload,
    )


def run_rag_benchmark(fmt: str = "text") -> str:
    return build_rag_report().render(fmt)
