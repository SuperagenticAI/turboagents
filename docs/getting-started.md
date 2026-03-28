# Getting Started

TurboAgents is easiest to understand when you treat it as infrastructure, not a
framework. You install the package, run a few health checks, and then choose
the path that matches what you already have: a runtime, a retrieval stack, or a
benchmarking question.

If you are new to the project, the goal of this page is simple: get you from
install to one useful command without making you read the whole repository
first.

## Choose Your Path

| If you want to... | Start here |
| --- | --- |
| confirm the package is healthy | run `turboagents doctor` |
| inspect compression quality quickly | run `turboagents bench kv` |
| inspect retrieval quality quickly | run `turboagents bench rag` |
| explore local serving integration | run the MLX dry-run command |
| try a concrete retrieval adapter | run one of the example scripts |
| run the fuller benchmark harness | use the matrix scripts later in this guide |

## Install

Start with the core package:

```bash
uv add turboagents
```

Then add only the extras you actually need:

```bash
uv add "turboagents[mlx]"
uv add "turboagents[rag]"
uv add "turboagents[all]"
```

Use `mlx` if you care about local runtime work. Use `rag` if you care about
Chroma, FAISS, LanceDB, SurrealDB, or pgvector. Use `all` only if you want the
full local surface in one environment.

If you are working from the repository directly instead of installing from
PyPI:

```bash
uv sync
uv sync --extra rag
uv sync --extra rag --extra mlx
```

## First Five Minutes

If you only want to see whether TurboAgents is wired correctly, run these in
order:

```bash
turboagents doctor
turboagents bench kv
turboagents bench rag
turboagents serve --backend mlx --model mlx-community/Qwen3-0.6B-4bit --dry-run
```

What those commands tell you:

- `turboagents doctor` checks whether the local runtime and adapter dependencies
  are visible.
- `turboagents bench kv` shows the compression and reconstruction story.
- `turboagents bench rag` shows the retrieval-agreement story.
- `turboagents serve --dry-run` shows what the runtime command would look like
  without starting a server.

If those commands make sense to you, the package is already in a usable state.

## Core CLI

These are the commands most people touch first:

```bash
turboagents doctor
turboagents bench kv
turboagents bench rag
turboagents bench paper
turboagents compress --input vectors.npy --output vectors.npz --head-dim 128
```

Use `bench kv` when you care about payload size and reconstruction quality. Use
`bench rag` when you care about retrieval agreement. Use `bench paper` when you
want the paper-style synthetic comparison surface. Use `compress` when you want
to apply the quantization path to your own vector file.

## Start With A Real Example

The example scripts are intentionally small. They are meant to show the API
shape and the runtime contract without hiding everything behind a framework.

If you want the fastest package-level examples:

```bash
python3 examples/quickstart.py
python3 examples/bench_profiles.py
```

If you want retrieval examples:

```bash
python3 examples/faiss_turborag.py
python3 examples/chroma_turborag.py
```

If you want serving integration:

```bash
python3 examples/mlx_server_dry_run.py
```

## When You Want Bigger Benchmarks

The CLI benchmarks are the fastest starting point, but the repository also
includes a fuller benchmark harness for deeper runs.

```bash
uv sync --extra rag --extra mlx
uv run python scripts/run_benchmark_matrix.py \
  --output-dir benchmark-results/$(date +%Y%m%d-%H%M%S)
```

If you want the minimal long-context check as well:

```bash
uv run python scripts/benchmark_needle.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --context-tokens 2048 4096 8192 \
  --output benchmark-results/needle-$(date +%Y%m%d-%H%M%S).json
```

Use this path when you want fuller benchmark artifacts and markdown summaries,
not just the lightweight CLI outputs.

## Local Docs

For local preview:

```bash
uv run mkdocs serve -f mkdocs.local.yml
```

For a static build:

```bash
uv run mkdocs build
```

## Where To Go Next

After this page, most people should continue in one of these directions:

- [Adapters](adapters.md) if you want to choose a backend path
- [Examples](examples.md) if you want runnable code first
- [Benchmarks](benchmarks.md) if you want the current numbers
- [Architecture](architecture.md) if you want the runtime and retrieval layout
