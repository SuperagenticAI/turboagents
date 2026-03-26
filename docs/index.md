# turboagents

![turboagents logo](assets/logo.png)

**Turbocharge AI Agents with TurboQuant**

`turboagents` is a Python package for TurboQuant-style KV-cache and vector
compression.

It is being built as:

- a reusable quantization core
- a benchmark surface
- a set of engine adapters
- a set of TurboRAG adapters

## Current State

Already implemented:

- structured quant payloads with binary serialization
- Fast Walsh-Hadamard rotation
- PolarQuant-style angle/radius encoding
- seeded QJL-style residual sketch
- synthetic benchmark CLI
- real FAISS, LanceDB, and SurrealDB adapter surfaces
- pgvector client adapter surface
- MLX runtime/server wrapper
- llama.cpp runtime wrapper
- experimental vLLM runtime wrapper

Not finished yet:

- native production kernels
- large benchmark reproduction
- full long-context benchmark matrix
- live pgvector/Postgres validation on this machine

## Project Boundaries

- `reference/` is analysis-only
- code from `reference/` must not be copied into `turboagents`
- integrations should target current upstream libraries
