# Adapters

TurboAgents is strongest where it can sit beside an existing engine or vector
store and add compression or reranking without forcing a larger application
rewrite.

## Engine Adapters

### MLX

The MLX path is the most practical runtime integration today. TurboAgents can
detect installed `mlx_lm`, inspect whether native `kv_bits` support is present,
build `mlx_lm.server` commands, load a local model through the wrapper, and run
small local generations. It does not yet ship native TurboQuant MLX kernels,
so today it relies on MLX-LM's existing quantized KV path rather than replacing
it.

### llama.cpp

The llama.cpp adapter is a wrapper-first integration. It can discover local
executables, inspect cache-type support from help output, build server
commands, and fall back cleanly when turbo cache types are not advertised. It
does not yet ship a native TurboQuant llama.cpp backend in this repository, so
real end-to-end use still depends on an external turbo-enabled runtime.

### vLLM

The vLLM adapter is still explicitly experimental. It covers command
construction, plugin and environment wiring, and an entry-point scaffold for
`vllm.general_plugins`. It does not yet provide a native TurboQuant backend,
and the current upstream KV cache path remains FP8-centric.

## TurboRAG Adapters

### Chroma

The Chroma adapter is a real client integration, not a placeholder. It supports
persistent local collections, candidate generation through Chroma, a
TurboAgents rerank path when local vector state is available, and a raw Chroma
candidate fallback after reopen when rerank state is not loaded. This is not a
Chroma `Context-1` replacement. The right fit is Chroma retrieval with
TurboAgents rerank underneath an external search loop. The current package is
aligned to `chromadb 1.5.5`.

### FAISS

FAISS is the most complete retrieval path in the package today. It uses a real
`faiss-cpu` index, supports candidate search, and can do exact rerank on top of
the compressed retrieval pass.

### LanceDB

The LanceDB adapter is a local sidecar integration. It opens a real LanceDB
connection, generates candidates through LanceDB, and then applies the
TurboAgents rerank path on top.

### SurrealDB

The SurrealDB adapter uses the real async client, supports an embedded
`mem://` path for tests, uses HNSW-backed candidate search, and then applies
the TurboAgents rerank layer. It is still a sidecar integration rather than the
deeper Rust-core codec path. See the dedicated [SurrealDB page](surrealdb.md)
for more detail.

### pgvector

The pgvector adapter focuses on compatibility with PostgreSQL applications. It
includes `psycopg2` and pgvector client wiring, schema helpers, add/search
operations, and an in-memory fallback when no database is reachable. It is
currently slower than FAISS and LanceDB in the checked benchmark path, and it
is still a sidecar client integration rather than a native compressed index.

## Best Fit

If you want the most mature runtime path today, choose MLX. If you want the
most complete TurboRAG path, choose FAISS. If you already depend on Chroma,
LanceDB, SurrealDB, or PostgreSQL, TurboAgents fits best as a sidecar rerank
layer that lets you keep the existing store in place while measuring the
quality and latency tradeoffs directly.
