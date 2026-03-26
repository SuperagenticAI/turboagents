#!/usr/bin/env python3
"""Run a minimal Needle-style long-context evaluation and write JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from turboagents.bench.needle import run_needle_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model id or local path for mlx_lm.load")
    parser.add_argument(
        "--context-tokens",
        nargs="+",
        type=int,
        default=[2048, 4096, 8192],
        help="Prompt token targets for the long-context sweep",
    )
    parser.add_argument(
        "--insertion-fractions",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 0.9],
        help="Needle insertion positions as fractions of the context body",
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        type=float,
        default=[3.0, 3.5, 4.0],
        help="TurboAgents bit-width sweep",
    )
    parser.add_argument("--max-tokens", type=int, default=32, help="Generation length limit")
    parser.add_argument("--needle", default="turboagents-needle-1729")
    parser.add_argument("--output", type=Path, required=True, help="Path to write JSON results")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_needle_benchmark(
        model_path=args.model,
        context_tokens=args.context_tokens,
        insertion_fractions=args.insertion_fractions,
        bits_list=args.bits,
        max_tokens=args.max_tokens,
        needle=args.needle,
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
