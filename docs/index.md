# turboagents

<div class="hero-mark">
  <img src="assets/logo.png" alt="turboagents logo">
</div>

<div class="hero-panel">
  <div class="hero-kicker">Compression Infrastructure For Real Systems</div>
  <h1 class="hero-title">TurboQuant for agent runtimes and retrieval stacks</h1>
  <p class="hero-lead">
    <code>turboagents</code> is a Python package for TurboQuant-style KV-cache and vector compression. It is designed to sit under existing AI systems, not replace them.
  </p>
  <div class="hero-actions">
    <a class="hero-action primary" href="getting-started.html">Start With uv</a>
    <a class="hero-action secondary" href="benchmarks.html">See Benchmarks</a>
    <a class="hero-action secondary" href="adapters.html">Browse Adapters</a>
  </div>
  <div class="hero-grid">
    <div class="hero-card">
      <h3>Quant Core</h3>
      <p>Fast Walsh-Hadamard rotation, PolarQuant-style angle/radius encoding, seeded QJL-style residual sketch, and binary payload serialization.</p>
    </div>
    <div class="hero-card">
      <h3>Real Adapters</h3>
      <p>MLX, llama.cpp, experimental vLLM, plus Chroma, FAISS, LanceDB, SurrealDB, and pgvector retrieval surfaces.</p>
    </div>
    <div class="hero-card">
      <h3>Validated Benchmarks</h3>
      <p>Benchmark matrix, MLX sweep, live pgvector validation, and a minimal Needle-style long-context harness.</p>
    </div>
  </div>
</div>

<div class="signal-grid">
  <div class="signal-card">
    <strong>Perfect Top-10 Recall</strong>
    <span>Chroma and FAISS both held full top-10 retrieval accuracy on the validated benchmark sweep.</span>
  </div>
  <div class="signal-card">
    <strong>Strong PostgreSQL Path</strong>
    <span>pgvector reached 0.896875 top-10 recall at 4.0 bits in live PostgreSQL validation.</span>
  </div>
  <div class="signal-card">
    <strong>Best MLX Tradeoff</strong>
    <span>3.5 bits was the best quality and throughput balance in the 3B MLX benchmark run.</span>
  </div>
  <div class="signal-card">
    <strong>Reference Integration</strong>
    <span>SuperOptiX is the first full application integration with real demo and retrieval coverage.</span>
  </div>
</div>

## Quick Snapshot

| Surface | Current Evidence |
| --- | --- |
| Chroma | local adapter benchmark reached `recall@10 = 1.0` across the tested bit-width sweep |
| MLX | cached `3B` smoke test passes and the `3B` sweep identified `3.5` bits as the best current tradeoff |
| FAISS | `recall@10 = 1.0` across the tested `medium-rag` bit-width sweep |
| pgvector | live PostgreSQL `17` validation completed, with `recall@10 = 0.896875` at `4.0` bits |
| Needle | exact-match retrieval only held at insertion fraction `0.1`; not yet robust at `0.5` or `0.9` |

<div class="section-band">

Use it when you already have:

- an agent runtime that is hitting KV-cache or context limits
- a RAG stack with growing vector storage cost
- an inference layer built on MLX, llama.cpp, or vLLM
- a retrieval layer built on Chroma, FAISS, LanceDB, SurrealDB, or pgvector

</div>

## What TurboAgents Is

TurboAgents is being built as:

- a reusable quantization core
- a benchmark surface
- a set of engine adapters
- a set of TurboRAG adapters

It is not an agent framework. It is compression infrastructure for the systems
you already run.

## Why The Current Version Is Useful

This repository is already useful if you want to:

- test compressed retrieval behavior without writing your own benchmark harness
- compare FAISS, LanceDB, and pgvector quality/latency tradeoffs on the same synthetic workload
- prototype MLX-based compressed serving paths locally
- measure where the current long-context story breaks instead of assuming it works

## How To Use It

Most users should use TurboAgents in one of three ways:

### Under An Existing Agent Runtime

Keep your current agent framework and use TurboAgents to improve the runtime
under it.

Examples:

- MLX-based local agents
- llama.cpp-based local agents
- experimental vLLM-backed serving stacks

### Under An Existing RAG Stack

Keep your current application logic and use TurboAgents in the retrieval layer.

Examples:

- FAISS-backed local retrieval
- Chroma candidate search plus TurboAgents rerank
- LanceDB or SurrealDB candidate search plus TurboAgents rerank
- pgvector-backed retrieval in PostgreSQL applications

### As A Benchmark And Compression Tool

Start with:

- `turboagents doctor`
- `turboagents bench kv`
- `turboagents bench rag`
- `turboagents compress`
- `uv run python scripts/run_benchmark_matrix.py --output-dir benchmark-results/<run-id>`

That gives you a low-risk way to decide where deeper integration is worth it.

## Start Here

If you are evaluating the project quickly, use this order:

1. Read [Getting Started](getting-started.md) and install with `uv`.
2. Run the synthetic CLI benchmarks locally.
3. Read [Adapters](adapters.md) and [Examples](examples.md) to pick the backend path you actually need.
4. Read [Benchmarks](benchmarks.md) for the current benchmark results.
5. Read [Architecture](architecture.md) if you want the runtime and retrieval layout.

## Reference Integration

TurboAgents is designed to stay standalone, but the first full reference
integration is now SuperOptiX.

That integration currently proves:

- `turboagents-chroma` works as a SuperOptiX retrieval option
- `turboagents-lancedb` works end to end in the SuperOptiX LanceDB demo
- `turboagents-surrealdb` works end to end in the SuperOptiX OpenAI Agents and
  Pydantic AI demos

If you want the end-to-end application story rather than the package-only API,
read the SuperOptiX TurboAgents guide after this page.

## Included In This Release

- structured quant payloads with binary serialization
- Fast Walsh-Hadamard rotation
- PolarQuant-style angle/radius encoding
- seeded QJL-style residual sketch
- synthetic benchmark CLI
- real Chroma, FAISS, LanceDB, and SurrealDB adapter surfaces
- pgvector client adapter surface
- MLX runtime and server wrapper
- llama.cpp runtime wrapper
- experimental vLLM runtime wrapper
- reproducible benchmark harness with checked-in result artifacts
- minimal Needle-style long-context evaluation harness
