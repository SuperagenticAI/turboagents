# Getting Started

This page is the shortest path from install to a real command. If you only want
to confirm that the package is wired correctly, you should be able to go from
zero to a working benchmark or dry-run serve command in a few minutes.

## Install

Start with the core package:

```bash
uv add turboagents
```

Then add extras only for the path you actually need:

```bash
uv add "turboagents[mlx]"
uv add "turboagents[rag]"
uv add "turboagents[all]"
```

If you are working from the repository directly:

```bash
uv sync
uv sync --extra rag
uv sync --extra rag --extra mlx
```

## First Five Minutes

If you only want to verify that the project is healthy, run:

```bash
turboagents doctor
turboagents bench kv
turboagents bench rag
turboagents serve --backend mlx --model mlx-community/Qwen3-0.6B-4bit --dry-run
```

These are the core commands most users will touch first:

```bash
turboagents doctor
turboagents bench kv
turboagents bench rag
turboagents bench paper
turboagents compress --input vectors.npy --output vectors.npz --head-dim 128
```

## Full Benchmark Workflow

When you want the larger benchmark matrix rather than the synthetic CLI checks,
use the scripts directly:

```bash
uv sync --extra rag --extra mlx
uv run python scripts/run_benchmark_matrix.py --output-dir benchmark-results/$(date +%Y%m%d-%H%M%S)
uv run python scripts/benchmark_needle.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --context-tokens 2048 4096 8192 --output benchmark-results/needle-$(date +%Y%m%d-%H%M%S).json
```

## Example Scripts

The example scripts are intentionally small. They are meant to show the shape
of the API and the runtime contract, not hide everything behind a framework.

```bash
python3 examples/quickstart.py
python3 examples/bench_profiles.py
python3 examples/faiss_turborag.py
python3 examples/chroma_turborag.py
python3 examples/mlx_server_dry_run.py
```

## Docs

For local docs preview:

```bash
uv run mkdocs serve -f mkdocs.local.yml
```

Build the static site:

```bash
uv run mkdocs build
```

If you are deciding where to go next, these are the useful follow-up pages:

- [Benchmarks](benchmarks.md)
- [Architecture](architecture.md)
- [Adapters](adapters.md)
