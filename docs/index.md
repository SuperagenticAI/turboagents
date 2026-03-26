# turboagents

![turboagents logo](assets/logo.png)

**Turbocharge AI Agents with TurboQuant**

`turboagents` is a Python package for TurboQuant-style KV-cache and vector
compression.

It is designed to sit under existing AI systems, not replace them.

Use it when you already have:

- an agent runtime that is hitting KV-cache or context limits
- a RAG stack with growing vector storage cost
- an inference layer built on MLX, llama.cpp, or vLLM
- a retrieval layer built on FAISS, LanceDB, SurrealDB, or pgvector

## What TurboAgents Is

TurboAgents is being built as:

- a reusable quantization core
- a benchmark surface
- a set of engine adapters
- a set of TurboRAG adapters

It is not an agent framework. It is compression infrastructure for the systems
you already run.

## How To Use It

Most users should use TurboAgents in one of three ways:

### Under An Existing Agent Runtime

Keep your current agent framework and use TurboAgents to improve the runtime
under it.

Examples:

- MLX-based local agents
- llama.cpp-based local agents
- experimental vLLM-backed serving stacks

### Under An Existing RAG Stack

Keep your current application logic and use TurboAgents in the retrieval layer.

Examples:

- FAISS-backed local retrieval
- LanceDB or SurrealDB candidate search plus TurboAgents rerank
- pgvector-backed retrieval in PostgreSQL applications

### As A Benchmark And Compression Tool

Start with:

- `turboagents doctor`
- `turboagents bench kv`
- `turboagents bench rag`
- `turboagents compress`

That gives you a low-risk way to decide where deeper integration is worth it.

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
