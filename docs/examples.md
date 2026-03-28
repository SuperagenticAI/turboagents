# Examples

Current package examples:

- `examples/quickstart.py`
  - basic quantize / serialize / dequantize flow
- `examples/bench_profiles.py`
  - prints the current KV, RAG, and paper-style synthetic benchmark reports
- `examples/mlx_server_dry_run.py`
  - shows the MLX server command TurboAgents builds for local serving
- `examples/faiss_turborag.py`
  - uses the live FAISS-backed TurboRAG adapter on a small local dataset
- `examples/chroma_turborag.py`
  - uses the Chroma-backed TurboRAG adapter on a small local dataset

## Reference Integration Examples

If you want the fully integrated application path rather than just package
examples, use the SuperOptiX reference integration:

- `rag_lancedb_demo`
  - validates `turboagents-lancedb` through a real SuperOptiX demo flow
- `rag_surrealdb_openai_demo`
  - validates `turboagents-surrealdb` with the OpenAI Agents runtime in SuperOptiX
- `rag_surrealdb_pydanticai_demo`
  - validates `turboagents-surrealdb` with the Pydantic AI runtime in SuperOptiX

The matching guide lives in the SuperOptiX docs:

- `https://superagenticai.github.io/superoptix/guides/turboagents-integration/`

## Recommended Starting Point

Start with:

- synthetic CLI benchmarks
- FAISS or Chroma adapter examples
- MLX dry-run command generation
- the SuperOptiX reference demos when you want end-to-end application coverage
