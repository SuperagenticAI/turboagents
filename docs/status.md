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
- `uv run mkdocs build` succeeds
- cached MLX `3B` smoke test works on:
  - `mlx-community/Llama-3.2-3B-Instruct-4bit`
- FAISS-backed TurboRAG example runs locally
- SurrealDB embedded mode tests pass locally
- reproducible higher-memory benchmark harness is now checked in under `scripts/` and `benchmarks/`
- full benchmark matrix run completed on the 128 GB Mac:
  - MLX sweep completed on `mlx-community/Llama-3.2-3B-Instruct-4bit`
  - FAISS adapter reached `recall@10 = 1.0` across tested bit-widths on `medium-rag`
  - LanceDB adapter reached `recall@10` in the `0.70` to `0.75` range on `medium-rag`
  - live PostgreSQL `17` + `pgvector` validation completed locally
  - pgvector reached `recall@10 = 0.896875` at `4.0` bits on `medium-rag`
  - minimal Needle-style long-context sweep completed on the 128 GB Mac
  - exact-match retrieval held only for the easy `0.1` insertion position and did not hold at `0.5` or `0.9`

## Not Finished Yet

- final paper-faithful production math
- true native engine kernels
- true upstream engine patches
- LongBench / larger benchmark datasets and a stronger long-context benchmark matrix
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
