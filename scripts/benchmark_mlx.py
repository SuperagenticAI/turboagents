#!/usr/bin/env python3
"""Run a lightweight MLX benchmark sweep and write JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

from turboagents.engines import mlx
from turboagents.quant.config import SUPPORTED_BITS


def _token_count(tokenizer: object, text: str) -> int | None:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return int(len(encode(text)))
        except Exception:
            return None
    return None


def run_benchmark(
    *,
    model_path: str,
    prompt: str,
    bits_list: list[float],
    max_tokens: int,
) -> dict[str, object]:
    load_started = time.perf_counter()
    model, tokenizer = mlx.load(model_path, lazy=False)
    load_seconds = time.perf_counter() - load_started

    prompt_tokens = _token_count(tokenizer, prompt)
    runs: list[dict[str, object]] = []
    for bits in bits_list:
        started = time.perf_counter()
        output = mlx.generate(
            model,
            tokenizer,
            prompt,
            bits=bits,
            max_tokens=max_tokens,
            temp=0.0,
        )
        elapsed = time.perf_counter() - started
        completion_text = output if isinstance(output, str) else str(output)
        completion_tokens = _token_count(tokenizer, completion_text)
        tokens_per_second = (
            None if completion_tokens in (None, 0) else round(completion_tokens / elapsed, 4)
        )
        runs.append(
            {
                "bits": bits,
                "elapsed_seconds": round(elapsed, 4),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "completion_chars": len(completion_text),
                "tokens_per_second": tokens_per_second,
                "preview": completion_text[:200],
            }
        )

    return {
        "benchmark": "mlx",
        "model": model_path,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "load_seconds": round(load_seconds, 4),
        "prompt": prompt,
        "max_tokens": max_tokens,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model id or local path for mlx_lm.load")
    parser.add_argument(
        "--bits",
        nargs="+",
        type=float,
        default=list(SUPPORTED_BITS),
        help="TurboAgents bit-width sweep",
    )
    parser.add_argument(
        "--prompt",
        default="Summarize how KV-cache compression helps long-context inference in two sentences.",
        help="Prompt used for the benchmark run",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Generation length limit")
    parser.add_argument("--output", type=Path, required=True, help="Path to write JSON results")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_benchmark(
        model_path=args.model,
        prompt=args.prompt,
        bits_list=args.bits,
        max_tokens=args.max_tokens,
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
