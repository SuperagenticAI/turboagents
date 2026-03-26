"""Experimental vLLM adapter helpers.

This wrapper stays honest about current upstream reality:

- official vLLM documents FP8 KV-cache quantization today
- TurboQuant integration requires plugin/kernel work that is not upstreamed

So the adapter implemented here is a real runtime wrapper with discovery,
environment wiring, serve-command construction, and plugin registration
scaffolding. It does not claim native upstream TurboQuant support.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
from typing import Mapping

from turboagents.engines.base import AdapterStatus, EngineRuntime


DEFAULT_EXECUTABLE = "vllm"
DEFAULT_PLUGIN_NAME = "turboagents"
DEFAULT_KV_CACHE_DTYPE = "fp8"
SUPPORTED_UPSTREAM_KV_CACHE_DTYPES = (
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
)


@dataclass(frozen=True, slots=True)
class VllmInstallation:
    executable: str | None
    version: str | None
    import_available: bool
    plugin_entrypoint: str
    help_text: str
    supported_kv_cache_dtypes: tuple[str, ...]


def _read_help(executable: str | None) -> str:
    if not executable:
        return ""
    try:
        completed = subprocess.run(
            [executable, "serve", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


def _detect_supported_kv_cache_dtypes(help_text: str) -> tuple[str, ...]:
    if not help_text:
        return SUPPORTED_UPSTREAM_KV_CACHE_DTYPES
    if "--kv-cache-dtype" not in help_text:
        return tuple()
    return SUPPORTED_UPSTREAM_KV_CACHE_DTYPES


def discover_installation() -> VllmInstallation:
    executable = shutil.which(DEFAULT_EXECUTABLE)
    import_available = importlib.util.find_spec("vllm") is not None
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = None
    help_text = _read_help(executable)
    return VllmInstallation(
        executable=executable,
        version=version,
        import_available=import_available,
        plugin_entrypoint="turboagents",
        help_text=help_text,
        supported_kv_cache_dtypes=_detect_supported_kv_cache_dtypes(help_text),
    )


def status() -> AdapterStatus:
    installation = discover_installation()
    available = installation.executable is not None or installation.import_available
    details: list[str] = ["experimental"]
    if installation.version:
        details.append(f"v{installation.version}")
    if installation.executable:
        details.append("serve CLI detected")
    elif installation.import_available:
        details.append("Python package detected")
    else:
        details.append("not installed")
    details.append("upstream KV dtype still FP8-centric")
    return AdapterStatus(
        name="vllm",
        experimental=True,
        available=available,
        detail=", ".join(details),
    )


def resolve_kv_cache_dtype(
    bits: float,
    *,
    requested_dtype: str | None = None,
) -> tuple[str, bool]:
    """Return the upstream KV cache dtype we can safely request today."""
    if requested_dtype is not None:
        if requested_dtype not in SUPPORTED_UPSTREAM_KV_CACHE_DTYPES:
            raise ValueError(f"Unsupported vLLM kv_cache_dtype={requested_dtype!r}.")
        return requested_dtype, False
    # Experimental TurboQuant path still uses FP8 as the upstream storage type.
    return DEFAULT_KV_CACHE_DTYPE, True


def build_plugin_env(
    *,
    bits: float,
    mode: str = "safe",
    plugin_name: str = DEFAULT_PLUGIN_NAME,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    existing = [item for item in env.get("VLLM_PLUGINS", "").split(",") if item]
    if plugin_name not in existing:
        existing.append(plugin_name)
    env["VLLM_PLUGINS"] = ",".join(existing)
    env["TURBOAGENTS_VLLM_MODE"] = mode
    env["TURBOAGENTS_VLLM_BITS"] = str(bits)
    return env


def build_serve_command(
    model: str,
    *,
    bits: float = 3.5,
    mode: str = "safe",
    host: str = "127.0.0.1",
    port: int = 8000,
    installation: VllmInstallation | None = None,
    requested_dtype: str | None = None,
    extra_args: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    installation = installation or discover_installation()
    executable = installation.executable or DEFAULT_EXECUTABLE
    resolved_dtype, fallback_applied = resolve_kv_cache_dtype(bits, requested_dtype=requested_dtype)
    command = [
        executable,
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--kv-cache-dtype",
        resolved_dtype,
    ]
    if mode == "safe":
        command.extend(["--generation-config", "vllm"])
    if extra_args:
        command.extend(extra_args)
    env = build_plugin_env(bits=bits, mode=mode)
    if fallback_applied:
        env["TURBOAGENTS_VLLM_UPSTREAM_DTYPE"] = resolved_dtype
    return command, env


def launch_server(
    model: str,
    *,
    bits: float = 3.5,
    mode: str = "safe",
    host: str = "127.0.0.1",
    port: int = 8000,
    installation: VllmInstallation | None = None,
    requested_dtype: str | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen[str]:
    command, env = build_serve_command(
        model,
        bits=bits,
        mode=mode,
        host=host,
        port=port,
        installation=installation,
        requested_dtype=requested_dtype,
        extra_args=extra_args,
    )
    return subprocess.Popen(command, env=env, text=True)


def enable(
    model: str,
    *,
    bits: float = 3.5,
    mode: str = "safe",
    host: str = "127.0.0.1",
    port: int = 8000,
    installation: VllmInstallation | None = None,
    requested_dtype: str | None = None,
) -> EngineRuntime:
    command, env = build_serve_command(
        model,
        bits=bits,
        mode=mode,
        host=host,
        port=port,
        installation=installation,
        requested_dtype=requested_dtype,
    )
    resolved_dtype, fallback_applied = resolve_kv_cache_dtype(bits, requested_dtype=requested_dtype)
    return EngineRuntime(
        name="vllm",
        bits=bits,
        mode=mode,
        options={
            "requested_kv_cache_dtype": requested_dtype or "turboagents_experimental",
            "resolved_kv_cache_dtype": resolved_dtype,
            "fallback_applied": fallback_applied,
            "plugin_name": DEFAULT_PLUGIN_NAME,
            "server_command": command,
            "env": {
                "VLLM_PLUGINS": env["VLLM_PLUGINS"],
                "TURBOAGENTS_VLLM_MODE": env["TURBOAGENTS_VLLM_MODE"],
                "TURBOAGENTS_VLLM_BITS": env["TURBOAGENTS_VLLM_BITS"],
                "TURBOAGENTS_VLLM_UPSTREAM_DTYPE": env.get("TURBOAGENTS_VLLM_UPSTREAM_DTYPE", ""),
            },
        },
    )
