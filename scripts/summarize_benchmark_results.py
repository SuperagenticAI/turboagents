#!/usr/bin/env python3
"""Summarize benchmark matrix artifacts into Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _render_metric_table(title: str, payload: dict[str, Any] | None) -> list[str]:
    lines = [f"## {title}", ""]
    if not payload:
        lines.append("_Not available in this run._")
        lines.append("")
        return lines

    metric_payload = payload.get("payload", payload)
    lines.extend(["| Metric | Value |", "| --- | --- |"])
    for key, value in metric_payload.items():
        lines.append(f"| {key} | {_fmt(value)} |")
    lines.append("")
    return lines


def _render_summary(summary: dict[str, Any] | None) -> list[str]:
    lines = ["## Command Summary", ""]
    if not summary:
        lines.append("_No `summary.json` found._")
        lines.append("")
        return lines

    lines.extend(["| Artifact | Return Code | Seconds |", "| --- | --- | --- |"])
    for run in summary.get("runs", []):
        lines.append(
            f"| {run.get('artifact', '-')} | {run.get('returncode', '-')} | {_fmt(run.get('elapsed_seconds'))} |"
        )
    lines.append("")
    return lines


def _render_rag_adapter_results(payload: dict[str, Any] | None) -> list[str]:
    lines = ["## RAG Adapter Matrix", ""]
    if not payload:
        lines.append("_No `rag-adapters.json` found._")
        lines.append("")
        return lines

    for adapter, results in payload.get("results", {}).items():
        lines.append(f"### {adapter}")
        lines.append("")
        lines.extend(
            [
                "| Bits | Build Seconds | Query Seconds | Recall@1 | Recall@10 | Status |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for bits, metrics in results.items():
            if "skipped" in metrics:
                lines.append(f"| {bits} | - | - | - | - | {metrics['skipped']} |")
            else:
                lines.append(
                    "| {bits} | {build} | {query} | {r1} | {r10} | ok |".format(
                        bits=bits,
                        build=_fmt(metrics.get("build_seconds")),
                        query=_fmt(metrics.get("query_seconds")),
                        r1=_fmt(metrics.get("recall_at_1")),
                        r10=_fmt(metrics.get("recall_at_10")),
                    )
                )
        lines.append("")
    return lines


def _render_mlx_results(payload: dict[str, Any] | None) -> list[str]:
    lines = ["## MLX Sweep", ""]
    if not payload:
        lines.append("_No `mlx-benchmark.json` found._")
        lines.append("")
        return lines

    lines.append(f"- Model: `{payload.get('model', '-')}`")
    lines.append(f"- Load seconds: `{_fmt(payload.get('load_seconds'))}`")
    lines.append("")
    lines.extend(
        [
            "| Bits | Elapsed Seconds | Prompt Tokens | Completion Tokens | Tokens / Second |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for run in payload.get("runs", []):
        lines.append(
            "| {bits} | {elapsed} | {prompt} | {completion} | {tps} |".format(
                bits=_fmt(run.get("bits")),
                elapsed=_fmt(run.get("elapsed_seconds")),
                prompt=_fmt(run.get("prompt_tokens")),
                completion=_fmt(run.get("completion_tokens")),
                tps=_fmt(run.get("tokens_per_second")),
            )
        )
    lines.append("")
    return lines


def build_markdown(result_dir: Path) -> str:
    manifest = _load_json(result_dir / "manifest.json")
    summary = _load_json(result_dir / "summary.json")
    kv = _load_json(result_dir / "bench-kv.json")
    rag = _load_json(result_dir / "bench-rag.json")
    paper = _load_json(result_dir / "bench-paper.json")
    rag_adapters = _load_json(result_dir / "rag-adapters.json")
    mlx = _load_json(result_dir / "mlx-benchmark.json")

    lines = ["# Benchmark Summary", ""]
    if manifest:
        lines.append(f"- Output dir: `{manifest.get('output_dir', result_dir)}`")
        lines.append(f"- Platform: `{manifest.get('platform', '-')}`")
        lines.append(f"- Python: `{manifest.get('python', '-')}`")
        lines.append("")

    lines.extend(_render_summary(summary))
    lines.extend(_render_metric_table("Synthetic KV Report", kv))
    lines.extend(_render_metric_table("Synthetic RAG Report", rag))
    lines.extend(_render_metric_table("Synthetic Paper Report", paper))
    lines.extend(_render_rag_adapter_results(rag_adapters))
    lines.extend(_render_mlx_results(mlx))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Benchmark result directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional markdown output path; defaults to <result_dir>/summary.md",
    )
    args = parser.parse_args()

    output = args.output or (args.result_dir / "summary.md")
    markdown = build_markdown(args.result_dir)
    output.write_text(markdown, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
