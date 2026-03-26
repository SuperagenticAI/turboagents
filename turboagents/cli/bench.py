"""Benchmark CLI wrapper."""

from turboagents.bench.kv import run_kv_benchmark
from turboagents.bench.paper import run_paper_benchmark
from turboagents.bench.rag import run_rag_benchmark


def run(target: str, fmt: str = "text") -> str:
    if target == "kv":
        return run_kv_benchmark(fmt)
    if target == "rag":
        return run_rag_benchmark(fmt)
    if target == "paper":
        return run_paper_benchmark(fmt)
    raise ValueError(f"Unknown benchmark target: {target}")
