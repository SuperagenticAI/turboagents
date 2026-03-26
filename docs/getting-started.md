# Getting Started

## Install

Core package:

```bash
pip install -e .
```

Useful extras:

```bash
pip install -e ".[bench]"
pip install -e ".[docs]"
pip install -e ".[mlx]"
pip install -e ".[rag]"
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
mkdocs serve
```

Build the static site:

```bash
mkdocs build
```
