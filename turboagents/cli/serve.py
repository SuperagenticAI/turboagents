"""Serve CLI wrapper."""

from __future__ import annotations

from turboagents.engines import llamacpp, mlx, vllm
from turboagents.proxy.server import run_server


def run(
    *,
    backend: str = "proxy",
    host: str = "127.0.0.1",
    port: int = 8000,
    model: str | None = None,
    bits: float = 3.5,
    dry_run: bool = False,
) -> str:
    if backend == "proxy":
        if dry_run:
            return f"Would start turboagents proxy on {host}:{port}"
        return run_server(host=host, port=port)
    if backend == "llamacpp":
        if not model:
            raise ValueError("--model is required for backend=llamacpp")
        runtime = llamacpp.enable(model, bits=bits, mode="safe")
        command = " ".join(runtime.options["server_command"])
        if dry_run:
            return command
        raise RuntimeError(
            "Direct llama.cpp process launching is intentionally dry-run only in the current CLI. "
            "Use the returned command to start the server explicitly."
        )
    if backend == "mlx":
        if not model:
            raise ValueError("--model is required for backend=mlx")
        runtime = mlx.enable_server(model, bits=bits, mode="safe", host=host, port=port)
        command = " ".join(runtime.options["server_command"])
        if dry_run:
            return command
        raise RuntimeError(
            "Direct MLX process launching is intentionally dry-run only in the current CLI. "
            "Use the returned command to start the server explicitly."
        )
    if backend == "vllm":
        if not model:
            raise ValueError("--model is required for backend=vllm")
        runtime = vllm.enable(model, bits=bits, mode="safe", host=host, port=port)
        env = runtime.options["env"]
        prefix = " ".join(
            [
                f"VLLM_PLUGINS={env['VLLM_PLUGINS']}",
                f"TURBOAGENTS_VLLM_MODE={env['TURBOAGENTS_VLLM_MODE']}",
                f"TURBOAGENTS_VLLM_BITS={env['TURBOAGENTS_VLLM_BITS']}",
            ]
        )
        command = " ".join(runtime.options["server_command"])
        if dry_run:
            return f"{prefix} {command}"
        raise RuntimeError(
            "Direct vLLM process launching is intentionally dry-run only in the current CLI. "
            "Use the returned command to start the server explicitly."
        )
    raise ValueError(f"Unknown backend: {backend}")
