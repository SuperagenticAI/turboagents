# Architecture

`turboagents` is one Python distribution with several internal product surfaces.

## Top-Level Structure

- `turboagents.quant`
- `turboagents.bench`
- `turboagents.engines`
- `turboagents.rag`
- `turboagents.proxy`
- `turboagents.cli`

## Quant Core

The current quantization path is:

1. seeded sign flips
2. Fast Walsh-Hadamard rotation
3. spherical angle/radius encoding
4. scalar angle quantization through generated codebooks
5. seeded QJL-style residual sketch
6. binary payload serialization

This gives the repo:

- a reusable codec surface
- a reproducible compressed representation
- one shared core used by benchmarks and adapters

## Bench Layer

The benchmark layer currently sits above the quant core and below the external
runtime integrations.

It provides:

- deterministic synthetic datasets
- multi-bit KV benchmark reports
- multi-bit RAG benchmark reports
- synthetic paper-style comparison reports

## Engine Layer

The engine layer is currently wrapper-first.

That means:

- build runtime commands
- detect installed capabilities
- expose a stable Python interface
- avoid pretending native TurboQuant kernels exist where they do not

This is intentional. The wrappers make it possible to validate the package
surface before deeper native integration work.

## TurboRAG Layer

The TurboRAG layer currently uses one of two patterns:

- native client/index integration where practical
- sidecar + rerank integration where full native compressed indexes do not exist

This is why:

- FAISS is already a real local path
- LanceDB and SurrealDB are real but still sidecar-oriented
- pgvector is client-backed but still waiting on live database validation

## Proxy and CLI

The CLI is the entrypoint users see first.

It currently covers:

- environment inspection
- benchmark execution
- compression demos
- serve command generation

The proxy is present as a baseline service layer and can evolve later into a
more complete product surface.
