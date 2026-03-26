# turboagents

<p align="center">
  <img src="assets/logo.png" alt="turboagents logo" width="220">
</p>

**Turbocharge AI Agents with TurboQuant**

`turboagents` is a single Python package for TurboQuant-style KV-cache and vector
compression. It is being built as independent compression infrastructure that
can be used standalone and integrated into SuperOptix.

## Status


- structured quantization payloads with binary serialization
- Fast Walsh-Hadamard rotation with cached sign patterns
- PolarQuant-style spherical angle/radius stage
- seeded QJL-style residual sketch
- synthetic benchmark CLI with KV, RAG, and paper-style reports
- real adapter surfaces for:
  - FAISS
  - LanceDB
  - SurrealDB
  - pgvector client adapter
  - MLX runtime/server wrapper
  - llama.cpp runtime wrapper
  - experimental vLLM runtime wrapper
- proxy/server baseline
- lightweight examples and docs

Still not finished:

- full paper-faithful production math
- native engine kernels for llama.cpp / MLX / vLLM
- live Postgres validation for pgvector on this machine
- large benchmark datasets and long-context benchmark matrix



## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[bench]"
pip install -e ".[docs]"
pip install -e ".[mlx]"
pip install -e ".[vllm]"
pip install -e ".[rag]"
pip install -e ".[all]"
```

## CLI

```bash
turboagents doctor
turboagents bench kv --format json
turboagents bench rag --format markdown
turboagents bench paper
turboagents compress --input vectors.npy --output vectors.npz --head-dim 128
turboagents serve --backend mlx --model mlx-community/Qwen3-0.6B-4bit --dry-run
turboagents serve --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --dry-run
```

## Examples

```bash
python3 examples/quickstart.py
python3 examples/bench_profiles.py
python3 examples/faiss_turborag.py
python3 examples/mlx_server_dry_run.py
```

## Current Local Validation

- cached MLX 3B smoke test succeeded on `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q` passes

## Attribution

See [ATTRIBUTION.md](ATTRIBUTION.md). This repository is not affiliated with
Google Research.
