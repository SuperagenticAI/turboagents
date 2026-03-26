"""Simple in-memory dashboard state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DashboardState:
    metrics: dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> None:
        self.metrics.update(kwargs)

    def snapshot(self) -> dict[str, Any]:
        return dict(self.metrics)


def describe_dashboard(state: DashboardState | None = None) -> str:
    if state is None:
        return "Dashboard is initialized."
    return "\n".join([f"{key}: {value}" for key, value in state.snapshot().items()])
