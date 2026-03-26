# SurrealDB

TurboAgents includes an active SurrealDB integration path as part of TurboRAG.

## Why SurrealDB

SurrealDB is a strong fit for TurboRAG because it already has:

- vector search support
- HNSW indexing
- document + graph + vector positioning that aligns well with agent systems

It is especially interesting because TurboAgents can add compression-oriented
retrieval behavior without requiring a completely different application stack.

## Current Integration Shape

The current implementation is a real client-side adapter built on the upstream
async Python client.

Implemented now:

- async SurrealDB client connection
- namespace and database selection
- collection creation
- HNSW-backed candidate retrieval
- TurboAgents payload storage alongside raw embeddings
- TurboAgents rerank over candidate results
- local embedded `mem://` testing path

This means the current path is already usable as:

- a development integration
- a demo path
- a sidecar/rerank retrieval layer

## What It Does Today

The current adapter works like this:

1. store the raw embedding in SurrealDB
2. store a serialized TurboAgents payload beside it
3. use SurrealDB HNSW for candidate retrieval
4. rerank candidates with the TurboAgents compressed representation

So the current value is:

- SurrealDB remains the database and candidate search layer
- TurboAgents adds compression-aware reranking and storage of compressed payloads

## Current Code Surface

The adapter lives in:

- `turboagents/rag/surrealdb.py`

Primary current interface:

```python
from turboagents.rag import TurboSurrealDB

store = TurboSurrealDB(
    url="mem://",
    namespace="testns",
    database="testdb",
    dim=128,
    bits=3.5,
    seed=0,
)

await store.create_collection("documents", dim=128)
await store.add(vectors, metadata=docs)
results = await store.search(query, k=5, rerank_top=16)
```

## What Is Not Finished Yet

The current SurrealDB integration is not yet:

- a native compressed index inside SurrealDB
- a Rust-core codec integration
- a new SurrealQL compression primitive

That deeper path is still future work.

## Why It Still Matters Now

Even before native integration, this adapter shows that TurboAgents is already
working toward SurrealDB support in a concrete way:

- real adapter
- real async client
- real HNSW candidate search
- real TurboAgents rerank path

So SurrealDB is not just a roadmap idea. It is an implemented, active
integration surface with more depth planned later.
