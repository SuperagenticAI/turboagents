# Implemented Surfaces

This page tracks what exists in the repository today, not the full long-term
plan.

## Snapshot

| Area | Implemented Now | Main Gap |
| --- | --- | --- |
| Quant core | rotation, PolarQuant-style encoding, residual sketch, binary payloads | paper-faithful production math |
| Benchmarks | synthetic CLI, MLX sweep, adapter matrix, minimal Needle harness | broader public benchmark suites |
| Engines | MLX, llama.cpp, experimental vLLM | native kernels and upstream patches |
| TurboRAG | Chroma, FAISS, LanceDB, SurrealDB, pgvector client adapter | native compressed indexes |

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
- checked-in 128 GB Mac benchmark harness
- minimal Needle-style long-context evaluation script

Still incomplete:

- LongBench / BEIR / MTEB integration
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
- real Chroma-backed adapter
- real LanceDB-backed adapter
- real SurrealDB-backed adapter
- pgvector client adapter

Still incomplete:

- native compressed index implementations

## Current Validation

Recent validated checks:

- full test suite passes
- cached MLX `3B` smoke test works locally
- FAISS adapter runs locally
- Chroma adapter smoke test passes locally against `chromadb 1.5.5`
- SurrealDB embedded mode runs locally
- live pgvector validation completed on the benchmark machine
- 128 GB Mac benchmark matrix completed
- minimal Needle sweep completed and documented
