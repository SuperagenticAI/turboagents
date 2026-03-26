"""Reporting helpers for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json


@dataclass(slots=True)
class Report:
    title: str
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"title": self.title, "payload": self.payload}, indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [f"## {self.title}", "", "| Metric | Value |", "| --- | --- |"]
        for key, value in self.payload.items():
            lines.append(f"| {key} | {value} |")
        return "\n".join(lines)

    def to_text(self) -> str:
        lines = [self.title]
        for key, value in self.payload.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def render(self, fmt: str = "text") -> str:
        if fmt == "text":
            return self.to_text()
        if fmt == "json":
            return self.to_json()
        if fmt == "markdown":
            return self.to_markdown()
        raise ValueError(f"Unsupported report format: {fmt}")
