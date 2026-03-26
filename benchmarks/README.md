# Benchmark Matrix

This directory defines the reproducible benchmark workflow for the higher-memory Mac.

## Goals

- keep the lightweight synthetic benchmark CLI for regression checks
- add repeatable MLX and RAG adapter validation runs
- store machine-readable artifacts that can be committed or summarized in docs

## Primary Entry Point

Install the needed extras first:

```bash
uv sync --extra rag --extra mlx
```

Run the benchmark matrix:

```bash
uv run python scripts/run_benchmark_matrix.py --output-dir benchmark-results/$(date +%Y%m%d-%H%M%S)
```

Include a real MLX run:

```bash
uv run python scripts/run_benchmark_matrix.py \
  --mlx-model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --output-dir benchmark-results/$(date +%Y%m%d-%H%M%S)
```

Include live pgvector validation:

```bash
uv run python scripts/run_benchmark_matrix.py \
  --pgvector-dsn postgresql://localhost/turboagents \
  --output-dir benchmark-results/$(date +%Y%m%d-%H%M%S)
```

## Produced Artifacts

- `manifest.json`: exact commands used for the run
- `summary.json`: exit codes and elapsed time
- `doctor.txt`: environment snapshot
- `bench-kv.json`: synthetic KV benchmark report
- `bench-rag.json`: synthetic RAG benchmark report
- `bench-paper.json`: synthetic paper-style report
- `rag-adapters.json`: FAISS / LanceDB / optional pgvector or SurrealDB results
- `mlx-benchmark.json`: optional MLX generation sweep
- `needle-benchmark.json`: optional minimal long-context Needle-style evaluation
- `summary.md`: generated markdown summary from the artifact set

Generate the markdown summary:

```bash
uv run python scripts/summarize_benchmark_results.py benchmark-results/<run-id>
```

Run the minimal long-context Needle harness directly:

```bash
uv run python scripts/benchmark_needle.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --context-tokens 2048 4096 8192 \
  --output benchmark-results/needle-$(date +%Y%m%d-%H%M%S).json
```

## Remaining Gaps After This Harness

- LongBench / Needle / BEIR / MTEB integration
- end-to-end serving latency matrix
- native engine kernels
- paper-faithful production math validation

This harness closes the operational gap for the 128 GB Mac. It does not claim that the full research benchmark stack already exists in-tree.
