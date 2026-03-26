"""turboagents command line entrypoint."""

from __future__ import annotations

import argparse

from turboagents.cli import bench, compress, doctor, serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turboagents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor")

    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument("target", choices=["kv", "rag", "paper"])
    bench_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
    )

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--backend", choices=["proxy", "llamacpp", "mlx", "vllm"], default="proxy")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--model")
    serve_parser.add_argument("--bits", type=float, default=3.5)
    serve_parser.add_argument("--dry-run", action="store_true")

    compress_parser = subparsers.add_parser("compress")
    compress_parser.add_argument("--input")
    compress_parser.add_argument("--output")
    compress_parser.add_argument("--bits", type=float, default=3.5)
    compress_parser.add_argument("--head-dim", type=int, default=128)
    compress_parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        print(doctor.run())
        return 0
    if args.command == "bench":
        print(bench.run(args.target, fmt=args.format))
        return 0
    if args.command == "serve":
        print(
            serve.run(
                backend=args.backend,
                host=args.host,
                port=args.port,
                model=args.model,
                bits=args.bits,
                dry_run=args.dry_run,
            )
        )
        return 0
    if args.command == "compress":
        print(
            compress.run(
                input_path=args.input,
                output_path=args.output,
                bits=args.bits,
                head_dim=args.head_dim,
                seed=args.seed,
            )
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
