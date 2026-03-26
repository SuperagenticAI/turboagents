# turboagents -- Implementation Plan

**By Superagentic AI**  
**Date:** March 25, 2026  
**Package:** `pip install turboagents`

## 1. Purpose

`turboagents` is a single Python package that implements the TurboQuant algorithm family for:

- KV cache compression in LLM inference engines
- Vector store compression for RAG
- Production tooling such as benchmarks, proxy serving, and hardware-aware recommendations

It is **not** an agent framework. It is compression infrastructure that can be used standalone and integrated into SuperOptix as a dependency.

## 2. Non-Negotiables

- The implementation must be an independent open-source implementation of the algorithms described in the TurboQuant paper and related prior work.
- The `reference/` directory is for **analysis only**.
- Code in `reference/` must **not** be copied into `turboagents`.
- Engine and database integrations must target the **latest supported upstream libraries** when implementation begins.
- The project must clearly distinguish:
  - paper-faithful core behavior
  - engine-specific adapter logic
  - experimental integrations

## 3. Packaging Strategy

Version 1 ships as **one Python distribution** with clear internal module boundaries and optional extras.

Why one package:

- simpler install story
- simpler docs and branding
- easier adoption for early users
- still compatible with later splitting if needed

Why not one dependency blob:

- engine and database integrations are heavy
- most users will only need a subset
- optional extras keep installs manageable

Planned extras:

- `turboagents[mlx]`
- `turboagents[llamacpp]`
- `turboagents[vllm]`
- `turboagents[rag]`
- `turboagents[proxy]`
- `turboagents[bench]`
- `turboagents[all]`

## 4. Package Layout

```text
turboagents/
├── turboagents/
│   ├── __init__.py
│   ├── quant/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── types.py
│   │   ├── hadamard.py
│   │   ├── codebooks.py
│   │   ├── polar.py
│   │   ├── qjl.py
│   │   ├── pipeline.py
│   │   └── context.py
│   ├── bench/
│   │   ├── __init__.py
│   │   ├── kv.py
│   │   ├── rag.py
│   │   ├── paper.py
│   │   ├── report.py
│   │   └── datasets.py
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── llamacpp.py
│   │   ├── mlx.py
│   │   └── vllm.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── faiss.py
│   │   ├── pgvector.py
│   │   ├── lancedb.py
│   │   └── surrealdb.py
│   ├── proxy/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   └── dashboard.py
│   └── cli/
│       ├── __init__.py
│       ├── main.py
│       ├── doctor.py
│       ├── bench.py
│       ├── serve.py
│       └── compress.py
├── tables/
├── tests/
├── examples/
├── docs/
├── pyproject.toml
├── README.md
├── LICENSE
└── ATTRIBUTION.md
```

## 5. Core Algorithm Scope

### 5.1 Core Quantization Pipeline

Implement the practical paper path:

1. Fast Walsh-Hadamard Transform with seeded sign flips
2. PolarQuant stage for angle/radius representation
3. Lloyd-Max scalar quantization using precomputed tables
4. QJL residual correction for unbiased inner-product estimation

### 5.2 Core Constraints

- Use the practical fast rotation path, not dense QR-based rotation matrices
- Treat the core as the trust anchor for every adapter and benchmark
- Keep the initial implementation correct and inspectable before optimizing kernels

### 5.3 Public Python Surface

Planned high-level API:

- `Config(bits=3.5, head_dim=128, seed=42, mode="mse" | "prod")`
- `quantize(vector, config)`
- `dequantize(compressed, config)`
- `inner_product(query, compressed, config)`
- `ContextCalculator(...)`

### 5.4 Core Test Targets

- Hadamard transform correctness
- Quantize/dequantize roundtrip sanity
- Inner-product estimator sanity
- Distortion checks against paper-level expectations
- Context estimation math sanity

## 6. Execution Modes

### 6.1 Safe Mode

Ship first.

- dequantize to standard representation
- call the engine's standard attention path
- correctness-first
- easier to validate and benchmark

### 6.2 Fused Mode

Ship second.

- compute directly on compressed KV or compressed vector forms where practical
- engine-specific optimization work
- higher performance potential
- more validation risk

## 7. Benchmarking First

Benchmarks are a product surface, not a docs appendix.

### 7.1 KV Benchmark Metrics

- memory saved
- effective max context
- TTFT
- decode throughput
- decode latency p50 and p99
- quality metrics such as Needle and LongBench
- rotation overhead

### 7.2 RAG Benchmark Metrics

- recall@10
- recall@100
- indexing time
- query latency
- memory footprint
- streaming insert performance

### 7.3 Benchmark Commands

Planned commands:

- `turboagents bench kv`
- `turboagents bench rag`
- `turboagents bench paper`

## 8. Engine Adapter Plan

### 8.1 llama.cpp

Priority: first-class.

Reasons:

- active ecosystem experimentation
- clear KV-cache seam
- strong value for safe-mode first

Implementation order:

1. safe mode wrapper
2. benchmark integration
3. fused path later

### 8.2 MLX

Priority: first-class.

Reasons:

- strong Apple Silicon story
- practical value for long-context local inference

Implementation order:

1. wrapper/monkey-patch path
2. benchmark integration
3. optimized Metal path later

### 8.3 vLLM

Priority: experimental.

Reasons:

- upstream quantized KV support is centered on FP8 today
- still useful because of plugin and serving ecosystem

Implementation order:

1. experimental adapter
2. benchmark only after functionality exists
3. no flagship positioning in v1

## 9. TurboRAG Plan

TurboRAG is a core pillar, not an optional afterthought.

### 9.1 Positioning

TurboRAG should be positioned as:

- zero-training
- online/data-oblivious
- unbiased inner-product compression
- hybrid rerank friendly

Do not position it as "the only vector compression story." Existing systems already support PQ, scalar quantization, or RaBitQ-like paths. The opportunity is the TurboQuant/TurboRAG product layer and its operating point.

### 9.2 Adapter Order

1. `FAISS`
2. `pgvector`
3. `LanceDB`
4. `SurrealDB`
5. `Qdrant`, `Chroma`, `Milvus`, others later

### 9.3 Why This Order

`FAISS`

- best for validating ANN math cleanly
- least product surface friction

`pgvector`

- best practical SQL demo
- clear value for production teams

`LanceDB`

- best benchmark battlefield against built-in quantized indexes such as PQ and RaBitQ-family support

`SurrealDB`

- attractive product gap where native quantization does not appear built in
- promising but deeper than a lightweight wrapper

### 9.4 SurrealDB Guidance

SurrealDB is promising, but should start as experimental.

Rules:

- use the local `reference/surrealdb` tree only to understand integration seams
- do not copy code
- implement against current upstream SurrealDB APIs and actual extension points
- treat native upstream syntax changes as a later phase

### 9.5 LanceDB Guidance

LanceDB is benchmark-first.

Rules:

- compare against its native index families
- do not over-promise a deep custom integration before benchmarks justify it

## 10. CLI Plan

Planned commands:

- `turboagents doctor`
- `turboagents bench kv`
- `turboagents bench rag`
- `turboagents bench paper`
- `turboagents serve`
- `turboagents compress`

### 10.1 doctor

Responsibilities:

- detect available hardware
- detect available engines and extras
- estimate recommended bit-widths
- estimate context expansion

### 10.2 serve

Responsibilities:

- launch an OpenAI-compatible server
- connect to supported backends
- expose live metrics where practical

### 10.3 compress

Responsibilities:

- offline compression for supported vector store targets
- dry-run and benchmark-friendly flow

## 11. SuperOptix Boundary

TurboAgents stays independent.

TurboAgents owns:

- core quantization
- engine adapters
- TurboRAG adapters
- benchmarks
- proxy and CLI

SuperOptix owns:

- playbook config
- GEPA-based tuning
- framework orchestration
- observability

Installation model:

- `pip install turboagents`
- `pip install superoptix[turbo]`

TurboAgents must not absorb framework-orchestration logic that belongs in SuperOptix.

## 12. Build Phases

### Phase 1: Core + Bench

Target outcome:

- usable `turboagents.quant`
- usable benchmark CLI
- core tests
- paper-reproduction subset

Deliverables:

- `quant/config.py`
- `quant/types.py`
- `quant/hadamard.py`
- `quant/codebooks.py`
- `quant/polar.py`
- `quant/qjl.py`
- `quant/pipeline.py`
- `quant/context.py`
- `bench/paper.py`
- `bench/report.py`
- `cli/bench.py`
- `cli/doctor.py`

### Phase 2: Engine Adapters

Target outcome:

- safe-mode `llama.cpp`
- first-class `MLX`
- initial `serve`

Deliverables:

- `engines/llamacpp.py`
- `engines/mlx.py`
- `proxy/server.py`
- `cli/serve.py`

### Phase 3: TurboRAG P0

Target outcome:

- FAISS adapter
- pgvector adapter
- offline compress command

Deliverables:

- `rag/faiss.py`
- `rag/pgvector.py`
- `cli/compress.py`
- `bench/rag.py`

### Phase 4: Experimental and P1

Target outcome:

- LanceDB benchmark adapter
- SurrealDB experimental adapter
- vLLM experimental adapter

Deliverables:

- `rag/lancedb.py`
- `rag/surrealdb.py`
- `engines/vllm.py`

## 13. Risk Controls

### 13.1 Naming

- use `turboagents` as the package name
- avoid `turboquant` and `turboagent` as package names

### 13.2 Scope

- do not start with fused kernels
- do not start with every vector store at once
- do not start with engine forks

### 13.3 Packaging

- one distribution
- modular internals
- heavy integrations behind extras

### 13.4 Quality

- benchmark before marketing
- preserve a paper-faithful baseline
- label experimental paths clearly

## 14. What We Will Not Build in v1

- no agent framework
- no weight quantization
- no training or fine-tuning system
- no deep engine forks as the primary strategy
- no SuperOptix-specific GEPA logic inside `turboagents`

## 15. Immediate Next Step

The next implementation step after this plan is:

1. scaffold the package
2. implement `quant/config.py`, `types.py`, and `hadamard.py`
3. add initial tests
4. build the benchmark and doctor CLI
5. only then move into engine and TurboRAG adapters
