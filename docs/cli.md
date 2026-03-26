# CLI

`turboagents` currently exposes a small set of top-level commands.

## doctor

Print the local environment and adapter availability.

```bash
turboagents doctor
```

Current output includes:

- platform and Python version
- optional package presence
- adapter summaries for:
  - llama.cpp
  - MLX
  - vLLM

## bench

Benchmark surfaces:

```bash
turboagents bench kv
turboagents bench rag
turboagents bench paper
```

Formats:

```bash
turboagents bench kv --format text
turboagents bench kv --format json
turboagents bench rag --format markdown
```

Targets:

- `kv`: synthetic KV-style reconstruction metrics across bit-widths
- `rag`: synthetic retrieval metrics across bit-widths
- `paper`: synthetic paper-style MSE / cosine comparison

## serve

Serve-related wrappers:

```bash
turboagents serve --backend proxy
turboagents serve --backend mlx --model mlx-community/Qwen3-0.6B-4bit --dry-run
turboagents serve --backend llamacpp --model model.gguf --dry-run
turboagents serve --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --dry-run
```

Backends:

- `proxy`
- `mlx`
- `llamacpp`
- `vllm`

The current CLI intentionally keeps real backend launching conservative. Dry-run
mode is the primary path for command construction.

## compress

Compress a local `.npy` vector file into serialized payloads:

```bash
turboagents compress \
  --input vectors.npy \
  --output vectors.npz \
  --bits 3.5 \
  --head-dim 128 \
  --seed 0
```

Current scope:

- local file input/output
- serialized payload generation
- useful as a codec/demo path
