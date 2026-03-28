# Adapters

This page summarizes the adapter surfaces available in TurboAgents.

## Engine Adapters

### MLX

Available today:

- detect installed `mlx_lm`
- detect native `kv_bits` support
- build `mlx_lm.server` commands
- load a local model through the wrapper
- run small local smoke generations

Notes:

- no native TurboQuant MLX kernels yet
- relies on MLX-LM's existing quantized KV path

### llama.cpp

Available today:

- discover local executables
- inspect cache-type support from help output
- build server commands
- fall back cleanly when turbo cache types are not advertised

Notes:

- no native TurboQuant llama.cpp integration in this repo yet
- depends on an external turbo-enabled runtime for real end-to-end use

### vLLM

Available today:

- command construction
- plugin/env wiring
- entry-point scaffold for `vllm.general_plugins`

Notes:

- no native TurboQuant backend integration yet
- upstream KV cache path remains FP8-centric today

## TurboRAG Adapters

### Chroma

Available today:

- real `chromadb` client connection
- persistent local collection support
- candidate generation through Chroma
- TurboAgents rerank path when local vector state is available
- raw Chroma candidate fallback after reopen when rerank state is not loaded

Notes:

- this is not a Chroma `Context-1` replacement
- the right fit today is Chroma retrieval with TurboAgents rerank underneath an
  external search loop

Aligned to `chromadb 1.5.5`.

### FAISS

Available today:

- real `faiss-cpu` index
- candidate search
- exact rerank

### LanceDB

Available today:

- real local LanceDB connection
- candidate generation through LanceDB
- TurboAgents rerank path

### SurrealDB

Available today:

- real async client
- embedded `mem://` test path
- HNSW-backed candidate search
- TurboAgents rerank path

Notes:

- this is not the deeper Rust-core codec integration yet

See the dedicated [SurrealDB page](surrealdb.md) for more detail.

### pgvector

Available today:

- psycopg2 + pgvector client wiring
- schema helpers
- add/search path
- in-memory fallback if no database is reachable

Notes:

- slower than FAISS and LanceDB on the current benchmark path
- still a sidecar client integration rather than a native compressed index

## Best Fit

- `MLX`: best current engine path
- `llama.cpp`: wrapper-ready, runtime-dependent
- `vLLM`: experimental
- `Chroma`: real local sidecar/rerank integration, useful fit for Chroma
  `Context-1` style search stacks
- `FAISS`: most complete current TurboRAG path
- `LanceDB` / `SurrealDB`: real sidecar/rerank integrations
- `pgvector`: live DB validation completed, but current path is slower than
  FAISS and LanceDB
