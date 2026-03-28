#!/usr/bin/env python3
"""Benchmark TurboRAG adapters and write JSON results."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import numpy as np

from turboagents.bench.datasets import make_vector_dataset
from turboagents.quant.config import SUPPORTED_BITS
from turboagents.rag import (
    TurboChroma,
    TurboFAISS,
    TurboLanceDB,
    TurboPgvector,
    TurboSurrealDB,
)


def _recall_at_k(pred: list[int], truth: list[int]) -> float:
    truth_set = set(truth)
    return float(sum(1 for item in pred if item in truth_set) / max(1, len(truth)))


def _exact_rankings(base: np.ndarray, queries: np.ndarray) -> list[tuple[list[int], list[int]]]:
    rankings: list[tuple[list[int], list[int]]] = []
    for query in queries:
        exact_scores = np.dot(base, query)
        rankings.append(
            (
                list(np.argsort(exact_scores)[::-1][:1]),
                list(np.argsort(exact_scores)[::-1][:10]),
            )
        )
    return rankings


def _summarize_searches(
    searches: list[list[dict[str, Any]]],
    truth: list[tuple[list[int], list[int]]],
    elapsed_seconds: float,
) -> dict[str, float]:
    recalls_1: list[float] = []
    recalls_10: list[float] = []
    for pred, (truth1, truth10) in zip(searches, truth):
        recalls_1.append(_recall_at_k([item["index"] for item in pred[:1]], truth1))
        recalls_10.append(_recall_at_k([item["index"] for item in pred[:10]], truth10))
    return {
        "query_seconds": round(elapsed_seconds, 6),
        "recall_at_1": round(float(np.mean(recalls_1)), 6),
        "recall_at_10": round(float(np.mean(recalls_10)), 6),
    }


def benchmark_faiss(base: np.ndarray, queries: np.ndarray, bits: float, seed: int) -> dict[str, float]:
    index = TurboFAISS(dim=int(base.shape[1]), bits=bits, seed=seed)
    build_started = time.perf_counter()
    index.add(base)
    build_seconds = time.perf_counter() - build_started

    truth = _exact_rankings(base, queries)
    query_started = time.perf_counter()
    searches = [index.search(query, k=10, rerank_top=50) for query in queries]
    query_seconds = time.perf_counter() - query_started
    payload = _summarize_searches(searches, truth, query_seconds)
    payload["build_seconds"] = round(build_seconds, 6)
    return payload


def benchmark_chroma(base: np.ndarray, queries: np.ndarray, bits: float, seed: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="turboagents-chroma-") as tmpdir:
        index = TurboChroma(
            path=tmpdir,
            collection_name="bench_vectors",
            dim=int(base.shape[1]),
            bits=bits,
            seed=seed,
        )
        build_started = time.perf_counter()
        index.create_collection("bench_vectors", base)
        build_seconds = time.perf_counter() - build_started

        truth = _exact_rankings(base, queries)
        query_started = time.perf_counter()
        searches = [index.search(query, k=10, rerank_top=50) for query in queries]
        query_seconds = time.perf_counter() - query_started
        payload = _summarize_searches(searches, truth, query_seconds)
        payload["build_seconds"] = round(build_seconds, 6)
        return payload


def benchmark_lancedb(base: np.ndarray, queries: np.ndarray, bits: float, seed: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="turboagents-lancedb-") as tmpdir:
        index = TurboLanceDB(uri=tmpdir, dim=int(base.shape[1]), bits=bits, seed=seed)
        build_started = time.perf_counter()
        index.create_table("bench_vectors", base)
        try:
            index.create_native_index()
        except Exception:
            pass
        build_seconds = time.perf_counter() - build_started

        truth = _exact_rankings(base, queries)
        query_started = time.perf_counter()
        searches = [index.search(query, k=10, rerank_top=50) for query in queries]
        query_seconds = time.perf_counter() - query_started
        payload = _summarize_searches(searches, truth, query_seconds)
        payload["build_seconds"] = round(build_seconds, 6)
        return payload


def benchmark_pgvector(
    base: np.ndarray,
    queries: np.ndarray,
    bits: float,
    seed: int,
    dsn: str,
) -> dict[str, float]:
    table = f"documents_b{str(bits).replace('.', '_')}"
    index = TurboPgvector(dsn=dsn, table=table, dim=int(base.shape[1]), bits=bits, seed=seed)
    conn = index.connect()
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    build_started = time.perf_counter()
    index.ensure_schema()
    index.add(base)
    build_seconds = time.perf_counter() - build_started

    truth = _exact_rankings(base, queries)
    query_started = time.perf_counter()
    searches = [index.search(query, k=10, rerank_top=50) for query in queries]
    query_seconds = time.perf_counter() - query_started
    normalized_searches = [
        [{**item, "index": int(item["index"]) - 1} for item in pred]
        for pred in searches
    ]
    payload = _summarize_searches(normalized_searches, truth, query_seconds)
    payload["build_seconds"] = round(build_seconds, 6)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    index.close()
    return payload


async def benchmark_surrealdb_async(
    base: np.ndarray,
    queries: np.ndarray,
    bits: float,
    seed: int,
    *,
    url: str,
    namespace: str,
    database: str,
) -> dict[str, float]:
    index = TurboSurrealDB(
        url=url,
        namespace=namespace,
        database=database,
        dim=int(base.shape[1]),
        bits=bits,
        seed=seed,
    )
    build_started = time.perf_counter()
    await index.create_collection("bench_vectors", dim=int(base.shape[1]))
    await index.add(base)
    build_seconds = time.perf_counter() - build_started

    truth = _exact_rankings(base, queries)
    query_started = time.perf_counter()
    searches = [await index.search(query, k=10, rerank_top=50) for query in queries]
    query_seconds = time.perf_counter() - query_started
    payload = _summarize_searches(searches, truth, query_seconds)
    payload["build_seconds"] = round(build_seconds, 6)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="medium-rag", help="Dataset name from turboagents.bench.datasets")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=["chroma", "faiss", "lancedb"],
        choices=["chroma", "faiss", "lancedb", "pgvector", "surrealdb"],
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        type=float,
        default=list(SUPPORTED_BITS),
        help="TurboAgents bit-width sweep",
    )
    parser.add_argument("--pgvector-dsn", help="Optional PostgreSQL DSN for pgvector validation")
    parser.add_argument("--surrealdb-url", help="Optional SurrealDB URL")
    parser.add_argument("--surrealdb-namespace", default="test")
    parser.add_argument("--surrealdb-database", default="test")
    parser.add_argument("--output", type=Path, required=True, help="Path to write JSON results")
    args = parser.parse_args()

    base, queries = make_vector_dataset(args.dataset)
    payload: dict[str, Any] = {
        "benchmark": "rag-adapters",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dataset": args.dataset,
        "vectors": int(base.shape[0]),
        "queries": int(queries.shape[0]),
        "dim": int(base.shape[1]),
        "results": {},
    }

    for adapter in args.adapters:
        adapter_results: dict[str, Any] = {}
        for bits in args.bits:
            try:
                if adapter == "faiss":
                    adapter_results[str(bits)] = benchmark_faiss(base, queries, bits, args.seed)
                elif adapter == "chroma":
                    adapter_results[str(bits)] = benchmark_chroma(base, queries, bits, args.seed)
                elif adapter == "lancedb":
                    adapter_results[str(bits)] = benchmark_lancedb(base, queries, bits, args.seed)
                elif adapter == "pgvector":
                    if not args.pgvector_dsn:
                        adapter_results[str(bits)] = {"skipped": "missing --pgvector-dsn"}
                    else:
                        adapter_results[str(bits)] = benchmark_pgvector(
                            base,
                            queries,
                            bits,
                            args.seed,
                            args.pgvector_dsn,
                        )
                elif adapter == "surrealdb":
                    if not args.surrealdb_url:
                        adapter_results[str(bits)] = {"skipped": "missing --surrealdb-url"}
                    else:
                        adapter_results[str(bits)] = asyncio.run(
                            benchmark_surrealdb_async(
                                base,
                                queries,
                                bits,
                                args.seed,
                                url=args.surrealdb_url,
                                namespace=args.surrealdb_namespace,
                                database=args.surrealdb_database,
                            )
                        )
            except Exception as exc:
                adapter_results[str(bits)] = {
                    "skipped": f"{adapter} benchmark unavailable",
                    "error": str(exc),
                }
        payload["results"][adapter] = adapter_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
