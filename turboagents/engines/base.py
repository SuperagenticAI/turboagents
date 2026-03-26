"""Common engine adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AdapterStatus:
    name: str
    experimental: bool = False
    available: bool = False
    detail: str = "not checked"


@dataclass(slots=True)
class EngineRuntime:
    name: str
    bits: float
    mode: str
    options: dict[str, Any]
