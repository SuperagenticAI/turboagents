# Examples

Current lightweight examples:

- `examples/quickstart.py`
  - basic quantize / serialize / dequantize flow
- `examples/bench_profiles.py`
  - prints the current KV, RAG, and paper-style synthetic benchmark reports
- `examples/mlx_server_dry_run.py`
  - shows the real MLX server command TurboAgents builds on this Mac
- `examples/faiss_turborag.py`
  - uses the live FAISS-backed TurboRAG adapter on a small local dataset

## Recommended Use On This Mac

Because this machine has limited RAM, prefer:

- synthetic benchmarks
- FAISS adapter smoke tests
- MLX dry-run command generation
- tiny 1B to 3B model smoke tests only

Save heavier long-context and large-model validation for the higher-memory Mac.
