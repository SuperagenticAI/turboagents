# Status

This page tracks what is implemented now, what has been validated locally, and
what is still incomplete.

## Implemented

- quantization core with:
  - Fast Walsh-Hadamard rotation
  - PolarQuant-style angle/radius encoding
  - seeded QJL-style residual sketch
  - binary compressed payload format
- CLI:
  - `doctor`
  - `bench kv`
  - `bench rag`
  - `bench paper`
  - `compress`
  - `serve`
- engine adapter surfaces:
  - MLX / MLX-LM
  - llama.cpp wrapper
  - experimental vLLM wrapper
- TurboRAG surfaces:
  - FAISS
  - LanceDB
  - SurrealDB
  - pgvector client adapter
- docs and examples

## Validated Locally

- full test suite passes
- `mkdocs build` succeeds
- cached MLX `3B` smoke test works on:
  - `mlx-community/Llama-3.2-3B-Instruct-4bit`
- FAISS-backed TurboRAG example runs locally
- SurrealDB embedded mode tests pass locally

## Not Finished Yet

- final paper-faithful production math
- true native engine kernels
- true upstream engine patches
- live Postgres validation for pgvector on this machine
- larger benchmark datasets and long-context benchmark matrix
- native compressed index implementations for databases

## Recommended Machine Split

Use this Mac for:

- package development
- synthetic benchmarks
- FAISS and adapter smoke tests
- small MLX model smoke tests

Use the higher-memory Mac for:

- large models
- long-context evaluation
- real benchmark matrix runs
- heavier MLX experiments
