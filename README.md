# turboagents

<p align="center">
  <img src="assets/logo.png" alt="turboagents logo" width="220">
</p>

<p align="center">
  <a href="https://pypi.org/project/turboagents/"><img src="https://img.shields.io/pypi/v/turboagents.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/turboagents/"><img src="https://img.shields.io/pypi/pyversions/turboagents.svg" alt="Python versions"></a>
  <a href="https://github.com/SuperagenticAI/turboagents/actions/workflows/ci.yml"><img src="https://github.com/SuperagenticAI/turboagents/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-black.svg" alt="License"></a>
</p>

**Turbocharge AI Agents with TurboQuant**

`turboagents` is a single Python package for TurboQuant-style KV-cache and vector
compression. It is being built as independent compression infrastructure that
can be used standalone and integrated into SuperOptix.

Repository: `https://github.com/SuperagenticAI/turboagents`

## Why It Exists

Most AI stacks do not need another agent framework. They need the memory and
retrieval layer underneath their existing agents to stop getting in the way.

`turboagents` is aimed at that layer:

- compress KV-cache payloads so local and server-side inference can hold more context
- compress vector payloads so retrieval systems can store and rerank more cheaply
- benchmark the quality, latency, and recall tradeoffs explicitly instead of hiding them
- integrate with runtimes and vector backends teams already use

## At A Glance

| Area | Current State |
| --- | --- |
| Quant core | Fast Walsh-Hadamard rotation, PolarQuant-style angle/radius stage, seeded QJL-style residual sketch, binary payloads |
| Engines | MLX wrapper, llama.cpp wrapper, experimental vLLM wrapper/plugin scaffold |
| Retrieval | Chroma, FAISS, LanceDB, SurrealDB, and pgvector client adapters |
| Benchmarks | Synthetic CLI, benchmark matrix, MLX sweep, adapter matrix, minimal Needle harness |
| Packaging | `uv`-first local workflow, docs, CI, release workflow, PyPI package |

## Quick Proof Points

The repository now has benchmark evidence, not just API surface:

- Chroma reached `recall@10 = 1.0` across the tested bit-width sweep in the local adapter benchmark
- FAISS reached `recall@10 = 1.0` across the tested bit-width sweep on `medium-rag`
- pgvector reached `recall@10 = 0.896875` at `4.0` bits on `medium-rag`
- the MLX `3B` sweep showed `3.5` bits as the best quality/performance operating point in that run
- the minimal Needle long-context run showed early-position retrieval, but not robust mid/late-position retrieval

That matters because the repo can now make narrower, defensible claims:

- it already works as compression infrastructure and benchmark tooling
- it does not yet justify broad long-context quality claims

## Fast Start

Install the package:

```bash
uv add turboagents
```

Install with useful extras:

```bash
uv add "turboagents[mlx]"
uv add "turboagents[rag]"
uv add "turboagents[all]"
```

Try the CLI first:

```bash
turboagents doctor
turboagents bench kv --format json
turboagents bench rag --format markdown
turboagents serve --backend mlx --model mlx-community/Qwen3-0.6B-4bit --dry-run
```

## Reference Integration

TurboAgents stays framework-agnostic, but the first full reference integration
is now in SuperOptiX.

That matters because the current validated story is not only package-level unit
tests. It also includes real SuperOptiX retrieval paths using TurboAgents under
framework runtimes.

Current reference-integration status:

- `turboagents-chroma` is wired into SuperOptiX and covered by focused runtime tests
- `turboagents-lancedb` is validated through the real `rag_lancedb_demo` flow
- `turboagents-surrealdb` is validated through the real SuperOptiX OpenAI Agents
  and Pydantic AI demo flows
- the DSPy SurrealDB path is still blocked by a local LiteLLM/Ollama compatibility
  issue, not by the TurboAgents retrieval layer itself

If you want the end-to-end integration story, start here after installing
TurboAgents:

- SuperOptiX integration guide:
  `https://superagenticai.github.io/superoptix/guides/turboagents-integration/`
- SuperOptiX LanceDB demo:
  `https://superagenticai.github.io/superoptix/examples/agents/rag-lancedb-demo/`
- SuperOptiX SurrealDB frameworks guide:
  `https://superagenticai.github.io/superoptix/examples/agents/surrealdb-frameworks-demo/`

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
- inference stacks built on MLX, llama.cpp, vLLM, Chroma, FAISS, LanceDB, SurrealDB, or pgvector
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
- use `TurboChroma` when you want Chroma candidate search plus TurboAgents rerank
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

- `TurboChroma` for Chroma candidate search + TurboAgents rerank
- `TurboLanceDB` for LanceDB candidate search + TurboAgents rerank
- `TurboSurrealDB` for SurrealDB candidate search + TurboAgents rerank
- `TurboPgvector` for PostgreSQL-backed storage and retrieval

## Chroma and Context-1

TurboAgents now includes a Chroma adapter aligned to `chromadb 1.5.5`.

The right integration model is:

- `Context-1` handles search policy and context management
- TurboAgents handles compressed retrieval and rerank
- Chroma retrieves candidates while TurboAgents reranks or compresses the
  working set under that loop

## Benchmarks Snapshot

Latest validated benchmark work:

| Surface | Result |
| --- | --- |
| Chroma | `recall@1 = 1.0`, `recall@10 = 1.0` across the tested sweep in the local adapter benchmark |
| MLX sweep | `3.5` bits was the best current quality/performance tradeoff on `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| FAISS | `recall@1 = 1.0`, `recall@10 = 1.0` across the tested sweep |
| LanceDB | `recall@10` landed in the `0.70` to `0.75` range on `medium-rag` |
| pgvector | `recall@10` improved monotonically up to `0.896875` at `4.0` bits |
| Needle | exact match held for insertion fraction `0.1`, but failed at `0.5` and `0.9` |

For the full numbers, see:

- [docs/benchmarks.md](docs/benchmarks.md)
- [docs/status.md](docs/status.md)
- [benchmark-results/20260326-128gb-run/summary.md](benchmark-results/20260326-128gb-run/summary.md)

## Docs Map

For the shortest path through this repo:

1. [docs/getting-started.md](docs/getting-started.md) for install and first commands
2. [docs/adapters.md](docs/adapters.md) for backend-specific retrieval surfaces
3. [docs/examples.md](docs/examples.md) for runnable local examples
4. [docs/benchmarks.md](docs/benchmarks.md) for validated benchmark numbers
5. [docs/status.md](docs/status.md) for what is implemented versus still incomplete

## Current Status


- structured quantization payloads with binary serialization
- Fast Walsh-Hadamard rotation with cached sign patterns
- PolarQuant-style spherical angle/radius stage
- seeded QJL-style residual sketch
- synthetic benchmark CLI with KV, RAG, and paper-style reports
- real adapter surfaces for:
  - Chroma
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
- larger benchmark datasets and stronger long-context evaluation
- production-strength long-context quality claims

## Install

```bash
uv add turboagents
```

Optional extras:

```bash
uv add "turboagents[mlx]"
uv add "turboagents[vllm]"
uv add "turboagents[rag]"
uv add "turboagents[all]"
```

For local development in this repository:

```bash
uv sync
uv sync --extra rag
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
python3 examples/chroma_turborag.py
python3 examples/mlx_server_dry_run.py
```

## Current Local Validation

- cached MLX 3B smoke test succeeded on `mlx-community/Llama-3.2-3B-Instruct-4bit`
- Chroma adapter smoke run succeeded on `chromadb 1.5.5`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest -q` passes

## Development

Common local commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest -q
uv run mkdocs serve
uv build
```

Benchmark harness commands:

```bash
uv sync --extra rag --extra mlx
uv run python scripts/run_benchmark_matrix.py --output-dir benchmark-results/$(date +%Y%m%d-%H%M%S)
uv run python scripts/benchmark_needle.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --context-tokens 2048 4096 8192 --output benchmark-results/needle-$(date +%Y%m%d-%H%M%S).json
```

Community and project health files:

- [CHANGELOG.md](CHANGELOG.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- [SUPPORT.md](SUPPORT.md)

## Attribution

See [ATTRIBUTION.md](ATTRIBUTION.md). This repository is not affiliated with
Google Research.
