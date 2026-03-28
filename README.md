# turboagents

<p align="center">
  <img src="assets/logo.png" alt="turboagents logo" width="220">
</p>

<p align="center">
  <a href="https://pypi.org/project/turboagents/"><img src="https://img.shields.io/pypi/v/turboagents.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/turboagents/"><img src="https://img.shields.io/pypi/pyversions/turboagents.svg" alt="Python versions"></a>
  <a href="https://github.com/SuperagenticAI/turboagents/actions/workflows/ci.yml"><img src="https://github.com/SuperagenticAI/turboagents/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://superagenticai.github.io/turboagents/"><img src="https://img.shields.io/badge/docs-live-black.svg" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-black.svg" alt="License"></a>
</p>

**Compression Infrastructure For Agent Runtimes And Retrieval Stacks**

`turboagents` is a single Python package for TurboQuant-style KV-cache and vector
compression. It is designed to sit underneath existing AI systems, not replace
them. If you already have an agent framework, a local inference stack, or a
RAG pipeline, TurboAgents gives you a way to add compression, reranking, and
benchmarking without rebuilding the rest of your application.

Docs: `https://superagenticai.github.io/turboagents/`  
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

## What You Can Use Today

The package is already useful in three common situations. If you are running
local agents, the MLX and llama.cpp wrappers give you a clean way to script and
inspect runtime paths. If you are running retrieval, the TurboRAG adapters let
you keep Chroma, FAISS, LanceDB, SurrealDB, or pgvector in place while adding a
compressed rerank layer. If you are still evaluating fit, the built-in
benchmarks give you a narrow and repeatable way to measure payload size,
reconstruction quality, and retrieval agreement before you change application
code.

The benchmark story is also real, not just conceptual. Chroma and FAISS both
held `recall@10 = 1.0` on the validated adapter sweep, pgvector reached
`recall@10 = 0.896875` at `4.0` bits, and the current MLX 3B run showed `3.5`
bits as the best quality and throughput tradeoff in that configuration. The
long-context story is intentionally narrower: the minimal Needle harness shows
early-position retrieval, but not robust mid- or late-position recall.

That is the right way to read this project today. TurboAgents is ready to use
as compression infrastructure and benchmark tooling. It is not yet making broad
claims about long-context quality or production-native kernels.

## Fast Start

Install the package with `uv`:

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

If you prefer to browse before installing, start with the public docs:

- `https://superagenticai.github.io/turboagents/`
- `https://superagenticai.github.io/turboagents/getting-started/`
- `https://superagenticai.github.io/turboagents/benchmarks/`

## Reference Integration

TurboAgents stays framework-agnostic, but the first full reference integration
is now in SuperOptiX.

That matters because the validated story is not limited to package-level tests.
It also includes real SuperOptiX retrieval paths using TurboAgents under
framework runtimes.

- `turboagents-chroma` is wired into SuperOptiX and covered by focused runtime tests
- `turboagents-lancedb` is validated through the real `rag_lancedb_demo` flow
- `turboagents-surrealdb` is validated through the real SuperOptiX OpenAI Agents
  and Pydantic AI demo flows

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

Most users approach TurboAgents in one of three ways.

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

If you want the full numbers and command paths, see:

- [docs/benchmarks.md](docs/benchmarks.md)
- [benchmark-results/20260326-128gb-run/summary.md](benchmark-results/20260326-128gb-run/summary.md)

## Docs Map

For the shortest path through the public docs:

1. [docs/getting-started.md](docs/getting-started.md) for install and first commands
2. [docs/adapters.md](docs/adapters.md) for backend-specific retrieval surfaces
3. [docs/examples.md](docs/examples.md) for runnable local examples
4. [docs/benchmarks.md](docs/benchmarks.md) for validated benchmark numbers
5. [docs/architecture.md](docs/architecture.md) for the runtime and retrieval layout

## Install And CLI Reference

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

## Core CLI

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

## Development

Common local commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest -q
uv run mkdocs serve -f mkdocs.local.yml
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
