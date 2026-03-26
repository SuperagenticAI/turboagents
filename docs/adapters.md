# Adapters

This page summarizes the current adapter surfaces and their maturity.

## Engine Adapters

### MLX

Status: implemented wrapper surface.

Current capabilities:

- detect installed `mlx_lm`
- detect native `kv_bits` support
- build `mlx_lm.server` commands
- load a local model through the wrapper
- run small local smoke generations

Current limits:

- no native TurboQuant MLX kernels yet
- relies on MLX-LM's existing quantized KV path

### llama.cpp

Status: implemented runtime wrapper surface.

Current capabilities:

- discover local executables
- inspect cache-type support from help output
- build server commands
- fall back cleanly when turbo cache types are not advertised

Current limits:

- no native TurboQuant llama.cpp integration in this repo yet
- depends on an external turbo-enabled runtime for real end-to-end use

### vLLM

Status: experimental wrapper surface.

Current capabilities:

- command construction
- plugin/env wiring
- entry-point scaffold for `vllm.general_plugins`

Current limits:

- no installed `vllm` runtime on this machine
- no native TurboQuant backend integration yet
- upstream KV cache path remains FP8-centric today

## TurboRAG Adapters

### FAISS

Status: real local adapter.

Current capabilities:

- real `faiss-cpu` index
- candidate search
- exact rerank

### LanceDB

Status: real local adapter.

Current capabilities:

- real local LanceDB connection
- candidate generation through LanceDB
- TurboAgents rerank path

### SurrealDB

Status: real local adapter.

Current capabilities:

- real async client
- embedded `mem://` test path
- HNSW-backed candidate search
- TurboAgents rerank path

Current limits:

- this is not the deeper Rust-core codec integration yet

See the dedicated [SurrealDB page](surrealdb.md) for more detail.

### pgvector

Status: real client adapter surface.

Current capabilities:

- psycopg2 + pgvector client wiring
- schema helpers
- add/search path
- in-memory fallback if no database is reachable

Current limits:

- no live Postgres runtime available on this machine during development

## Maturity Summary

- `MLX`: best current engine path on this Mac
- `llama.cpp`: wrapper-ready, runtime-dependent
- `vLLM`: experimental
- `FAISS`: most complete current TurboRAG path
- `LanceDB` / `SurrealDB`: real sidecar/rerank integrations
- `pgvector`: client surface present, live DB validation pending
