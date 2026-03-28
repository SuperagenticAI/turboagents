#!/usr/bin/env python3
"""Run the reproducible benchmark matrix for a higher-memory Mac."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def _run(command: list[str], *, cwd: Path) -> dict[str, object]:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    return {
        "command": command,
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 4),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results") / time.strftime("%Y%m%d-%H%M%S"),
        help="Directory for benchmark artifacts",
    )
    parser.add_argument("--mlx-model", help="Optional MLX model id/path for real generation benchmarking")
    parser.add_argument("--pgvector-dsn", help="Optional DSN for pgvector validation")
    parser.add_argument("--surrealdb-url", help="Optional SurrealDB URL")
    parser.add_argument("--surrealdb-namespace", default="test")
    parser.add_argument("--surrealdb-database", default="test")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[str, list[str]]] = [
        ("doctor.txt", ["uv", "run", "turboagents", "doctor"]),
        ("bench-kv.json", ["uv", "run", "turboagents", "bench", "kv", "--format", "json"]),
        ("bench-rag.json", ["uv", "run", "turboagents", "bench", "rag", "--format", "json"]),
        ("bench-paper.json", ["uv", "run", "turboagents", "bench", "paper", "--format", "json"]),
        (
            "rag-adapters.json",
            [
                "uv",
                "run",
                "python",
                "scripts/benchmark_rag_adapters.py",
                "--output",
                str(output_dir / "rag-adapters.json"),
            ],
        ),
    ]

    if args.pgvector_dsn:
        commands[-1][1].extend(["--adapters", "chroma", "faiss", "lancedb", "pgvector"])
        commands[-1][1].extend(["--pgvector-dsn", args.pgvector_dsn])
    elif args.surrealdb_url:
        commands[-1][1].extend(["--adapters", "chroma", "faiss", "lancedb", "surrealdb"])
        commands[-1][1].extend(
            [
                "--surrealdb-url",
                args.surrealdb_url,
                "--surrealdb-namespace",
                args.surrealdb_namespace,
                "--surrealdb-database",
                args.surrealdb_database,
            ]
        )

    if args.mlx_model:
        commands.append(
            (
                "mlx-benchmark.json",
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/benchmark_mlx.py",
                    "--model",
                    args.mlx_model,
                    "--output",
                    str(output_dir / "mlx-benchmark.json"),
                ],
            )
        )

    manifest = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "commands": [{"artifact": artifact, "command": command} for artifact, command in commands],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    summary: dict[str, object] = {"runs": []}
    for artifact, command in commands:
        result = _run(command, cwd=repo_root)
        if artifact.endswith(".txt") and result["returncode"] == 0:
            (output_dir / artifact).write_text(str(result["stdout"]), encoding="utf-8")
        elif artifact.endswith(".json") and result["returncode"] == 0 and artifact != "rag-adapters.json" and artifact != "mlx-benchmark.json":
            (output_dir / artifact).write_text(str(result["stdout"]), encoding="utf-8")
        run_record = {
            "artifact": artifact,
            "command": command,
            "returncode": result["returncode"],
            "elapsed_seconds": result["elapsed_seconds"],
            "stderr": result["stderr"],
        }
        summary["runs"].append(run_record)
        if result["returncode"] != 0:
            (output_dir / f"{artifact}.stderr.txt").write_text(str(result["stderr"]), encoding="utf-8")
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(run_record, indent=2), file=sys.stderr)
            return int(result["returncode"])

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
