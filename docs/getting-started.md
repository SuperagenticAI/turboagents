# Getting Started

This page is the fastest path to running the package locally with the current
`uv`-first workflow.

## Install

Core package:

```bash
uv add turboagents
```

Useful extras:

```bash
uv add "turboagents[mlx]"
uv add "turboagents[rag]"
uv add "turboagents[all]"
```

Local repository development:

```bash
uv sync
uv sync --extra rag
uv sync --extra rag --extra mlx
```

## First Five Minutes

If you only want to verify that the project is wired correctly, run:

```bash
turboagents doctor
turboagents bench kv
turboagents bench rag
turboagents serve --backend mlx --model mlx-community/Qwen3-0.6B-4bit --dry-run
```

## Basic Commands

```bash
turboagents doctor
turboagents bench kv
turboagents bench rag
turboagents bench paper
turboagents compress --input vectors.npy --output vectors.npz --head-dim 128
```

## Higher-Memory Benchmark Workflow

For the validated benchmark flow used on the 128 GB Mac:

```bash
uv sync --extra rag --extra mlx
uv run python scripts/run_benchmark_matrix.py --output-dir benchmark-results/$(date +%Y%m%d-%H%M%S)
uv run python scripts/benchmark_needle.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --context-tokens 2048 4096 8192 --output benchmark-results/needle-$(date +%Y%m%d-%H%M%S).json
```

## Example Scripts

```bash
python3 examples/quickstart.py
python3 examples/bench_profiles.py
python3 examples/faiss_turborag.py
python3 examples/mlx_server_dry_run.py
```

## Docs

Run the docs locally:

```bash
uv run mkdocs serve
```

Build the static site:

```bash
uv run mkdocs build
```

Useful follow-up pages:

- [Status](status.md)
- [Benchmarks](benchmarks.md)
- [Implemented](implemented.md)
