# Implemented Surfaces

This page tracks what exists in the repository today, not the full long-term
plan.

## Quant Core

Implemented:

- config and public API
- binary compressed payload format
- Fast Walsh-Hadamard rotation with cached sign patterns
- PolarQuant-style spherical angle/radius encoding
- seeded QJL-style residual sketch

Still incomplete:

- final paper-faithful Lloyd-Max tables
- final production residual estimator
- native fused kernels

## Benchmarks

Implemented:

- `turboagents bench kv`
- `turboagents bench rag`
- `turboagents bench paper`
- deterministic built-in synthetic datasets

Still incomplete:

- LongBench / Needle / BEIR / MTEB integration
- end-to-end serving benchmark matrix

## Engine Adapters

Implemented:

- MLX runtime wrapper
- MLX server command wrapper
- llama.cpp runtime wrapper
- experimental vLLM runtime wrapper and plugin scaffold

Still incomplete:

- native TurboQuant engine kernels
- upstream engine patches

## TurboRAG

Implemented:

- real FAISS-backed adapter
- real LanceDB-backed adapter
- real SurrealDB-backed adapter
- pgvector client adapter

Still incomplete:

- live Postgres validation on this machine
- native compressed index implementations

## Current Validation

Recent validated checks:

- full test suite passes
- cached MLX `3B` smoke test works locally
- FAISS adapter runs locally
- SurrealDB embedded mode runs locally
