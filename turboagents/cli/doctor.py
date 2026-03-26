"""Hardware and adapter inspection."""

from __future__ import annotations

import importlib.util
import platform
from shutil import which

import numpy as np

from turboagents.engines import llamacpp, mlx, vllm


def _module_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run() -> str:
    lines = [
        "System",
        f"  Platform: {platform.platform()}",
        f"  Python:   {platform.python_version()}",
        f"  NumPy:    {np.__version__}",
        "",
        "Tools",
        f"  git:      {'yes' if which('git') else 'no'}",
        "",
        "Optional packages",
        f"  mlx_lm:   {'yes' if _module_present('mlx_lm') else 'no'}",
        f"  vllm:     {'yes' if _module_present('vllm') else 'no'}",
        f"  faiss:    {'yes' if _module_present('faiss') else 'no'}",
        "",
        "Adapters",
    ]
    for module in (llamacpp, mlx, vllm):
        status = module.status()
        extra = " experimental" if status.experimental else ""
        lines.append(
            f"  {status.name}: {status.detail} [{('available' if status.available else 'pending')}{extra}]"
        )
    return "\n".join(lines)
