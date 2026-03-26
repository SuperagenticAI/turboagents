# Getting Started

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
```

## Basic Commands

```bash
turboagents doctor
turboagents bench kv
turboagents bench rag
turboagents bench paper
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
