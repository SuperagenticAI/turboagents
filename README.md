# turboagents

<p align="center">
  <img src="assets/logo.png" alt="turboagents logo" width="220">
</p>

**Turbocharge AI Agents with TurboQuant**

`turboagents` is a single Python package for TurboQuant-style KV-cache and vector
compression. It is being built as independent compression infrastructure that
can be used standalone and integrated into SuperOptix.

Repository: `https://github.com/SuperagenticAI/turboagents`

## What It Is

`turboagents` is not an agent framework. It is the compression layer you put
under existing AI agents, inference engines, and RAG stacks so they can:

- hold longer contexts
- use less KV-cache memory
- store more embeddings at lower cost
- benchmark quality and memory tradeoffs explicitly

Think of it as:

- `TurboQuant` for real systems
- `TurboRAG` for vector retrieval stacks
- adapters and tooling around existing engines instead of a replacement for them

## Who It Is For

`turboagents` is for teams and developers who already have:

- AI agents that hit memory limits on long prompts
- RAG systems with large embedding stores
- inference stacks built on MLX, llama.cpp, vLLM, FAISS, LanceDB, SurrealDB, or pgvector
- agent frameworks that need compression infrastructure, not another framework

## How To Use It

Most users should think about `turboagents` in three ways.

### 1. Add It Under An Existing Agent Runtime

If you already have an agent system, keep the agent layer and use `turboagents`
to improve the inference or memory layer under it.

Examples:

- use `turboagents.engines.mlx` for MLX-based local agents
- use `turboagents.engines.llamacpp` to build llama.cpp runtime commands
- use `turboagents.engines.vllm` as an experimental runtime wrapper

### 2. Add It Under An Existing RAG Stack

If you already have retrieval, keep your current application logic and add
TurboRAG where vectors are stored or searched.

Examples:

- use `TurboFAISS` when you want a local FAISS-backed retrieval path
- use `TurboLanceDB` or `TurboSurrealDB` when you want a sidecar/rerank integration
- use `TurboPgvector` when your application already depends on PostgreSQL

### 3. Use It As A Benchmark And Compression Tool

If you are still evaluating whether TurboQuant-style compression makes sense for
your stack, use the CLI first:

- `turboagents doctor`
- `turboagents bench kv`
- `turboagents bench rag`
- `turboagents compress`

That gives you a way to validate fit before deeper integration work.

## Usage Patterns

### Existing Agent + MLX

Use TurboAgents to build or validate the MLX runtime path, then keep your
existing agent code on top.

```bash
turboagents serve --backend mlx --model mlx-community/Qwen3-0.6B-4bit --dry-run
```

### Existing RAG + FAISS

Use TurboRAG as the retrieval layer while keeping your current ingestion and
agent orchestration logic.

```python
from turboagents.rag import TurboFAISS

index = TurboFAISS(dim=128, bits=3.5, seed=0)
index.add(vectors)
results = index.search(query, k=5, rerank_top=16)
```

### Existing Vector Database

If you already use a database-backed vector layer, TurboAgents should sit beside
that store first, then move deeper only if it proves useful.

Examples:

- `TurboLanceDB` for LanceDB candidate search + TurboAgents rerank
- `TurboSurrealDB` for SurrealDB candidate search + TurboAgents rerank
- `TurboPgvector` for PostgreSQL-backed storage and retrieval

## Current Status


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
