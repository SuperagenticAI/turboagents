# Examples

The package examples are intentionally small and direct. They are meant to show
the API shape, the runtime contract, and the retrieval flow without hiding
everything behind framework-specific abstractions.

`examples/quickstart.py` covers the core quantize, serialize, and dequantize
path. `examples/bench_profiles.py` prints the current KV, RAG, and paper-style
synthetic benchmark reports. `examples/mlx_server_dry_run.py` shows the MLX
server command TurboAgents builds for local serving. `examples/faiss_turborag.py`
and `examples/chroma_turborag.py` show the live FAISS-backed and Chroma-backed
TurboRAG adapters on small local datasets.

## Reference Integration Examples

If you want the fully integrated application path rather than package-only
examples, use the SuperOptiX reference integration. `rag_lancedb_demo`
validates `turboagents-lancedb` through a real SuperOptiX demo flow.
`rag_surrealdb_openai_demo` and `rag_surrealdb_pydanticai_demo` validate
`turboagents-surrealdb` under real framework runtimes inside SuperOptiX.

The matching guide lives in the SuperOptiX docs:

- `https://superagenticai.github.io/superoptix/guides/turboagents-integration/`

## Recommended Starting Point

If you are choosing a starting point, begin with the synthetic CLI benchmarks
and then move to the FAISS or Chroma examples. Use the MLX dry-run path when
you care about serving integration, and use the SuperOptiX reference demos when
you want full end-to-end application coverage rather than package-level API
examples.
