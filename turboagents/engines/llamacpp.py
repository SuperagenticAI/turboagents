"""llama.cpp external-runtime adapter helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from shutil import which
import subprocess
from typing import Iterable

from turboagents.engines.base import AdapterStatus, EngineRuntime


SERVER_CANDIDATES = (
    "llama-server",
    "server",
)
CLI_CANDIDATES = (
    "llama-cli",
    "main",
)
DEFAULT_FALLBACK_CACHE_TYPE = "f16"
KNOWN_CACHE_TYPES = {
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q4_0",
    "q4_1",
    "iq4_nl",
    "q5_0",
    "q5_1",
}


@dataclass(frozen=True, slots=True)
class LlamaCppInstallation:
    server_executable: str | None
    cli_executable: str | None
    help_text: str
    supported_cache_types: tuple[str, ...]

    @property
    def available(self) -> bool:
        return bool(self.server_executable or self.cli_executable)

    @property
    def supports_turbo_cache(self) -> bool:
        return any(value.startswith("turbo") for value in self.supported_cache_types)


def _read_help(executable: str | None) -> str:
    if not executable:
        return ""
    try:
        result = subprocess.run(
            [executable, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (result.stdout or "") + "\n" + (result.stderr or "")


def _extract_cache_types(help_text: str) -> tuple[str, ...]:
    discovered = set(KNOWN_CACHE_TYPES)
    if not help_text:
        return tuple(sorted(discovered))
    for match in re.finditer(r"allowed values:\s*([^\n]+)", help_text, flags=re.IGNORECASE):
        values = re.split(r"[,/\s]+", match.group(1).strip())
        for value in values:
            cleaned = value.strip(" .;()[]")
            if cleaned:
                discovered.add(cleaned)
    for match in re.finditer(r"\bturbo\d(?:\.\d)?\b", help_text):
        discovered.add(match.group(0))
    return tuple(sorted(discovered))


def discover_installation(
    *,
    server_executable: str | None = None,
    cli_executable: str | None = None,
) -> LlamaCppInstallation:
    server_path = server_executable or next(
        (path for name in SERVER_CANDIDATES if (path := which(name))), None
    )
    cli_path = cli_executable or next(
        (path for name in CLI_CANDIDATES if (path := which(name))), None
    )
    help_text = _read_help(server_path) or _read_help(cli_path)
    supported_cache_types = _extract_cache_types(help_text)
    return LlamaCppInstallation(
        server_executable=server_path,
        cli_executable=cli_path,
        help_text=help_text,
        supported_cache_types=supported_cache_types,
    )


def status() -> AdapterStatus:
    installation = discover_installation()
    if not installation.available:
        return AdapterStatus(
            name="llamacpp",
            available=True,
            detail="baseline configuration helper only; no local llama.cpp executable detected",
        )
    detail = (
        f"server={os.path.basename(installation.server_executable) if installation.server_executable else 'n/a'} "
        f"cli={os.path.basename(installation.cli_executable) if installation.cli_executable else 'n/a'} "
        f"turbo_cache={'yes' if installation.supports_turbo_cache else 'no'}"
    )
    return AdapterStatus(name="llamacpp", available=True, detail=detail)


def normalize_cache_type(bits: float) -> str:
    return f"turbo{bits}".replace(".0", "")


def resolve_cache_type(
    bits: float,
    installation: LlamaCppInstallation | None = None,
    *,
    allow_fallback: bool = True,
    fallback_cache_type: str = DEFAULT_FALLBACK_CACHE_TYPE,
) -> tuple[str, bool]:
    requested = normalize_cache_type(bits)
    installation = installation or discover_installation()
    if requested in installation.supported_cache_types or not installation.help_text:
        return requested, False
    if allow_fallback:
        return fallback_cache_type, True
    raise RuntimeError(
        f"Installed llama.cpp does not advertise support for cache type {requested!r}."
    )


def build_server_command(
    model_path: str,
    *,
    bits: float = 3.5,
    mode: str = "safe",
    host: str = "127.0.0.1",
    port: int = 8081,
    ctx_size: int | None = None,
    installation: LlamaCppInstallation | None = None,
    allow_fallback: bool = True,
    fallback_cache_type: str = DEFAULT_FALLBACK_CACHE_TYPE,
    extra_args: Iterable[str] | None = None,
) -> list[str]:
    installation = installation or discover_installation()
    executable = installation.server_executable or "llama-server"
    resolved_cache_type, _ = resolve_cache_type(
        bits,
        installation,
        allow_fallback=allow_fallback,
        fallback_cache_type=fallback_cache_type,
    )
    command = [
        executable,
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--cache-type-k",
        resolved_cache_type,
        "--cache-type-v",
        resolved_cache_type,
    ]
    if ctx_size is not None:
        command.extend(["--ctx-size", str(ctx_size)])
    if mode == "safe":
        command.extend(["--flash-attn"])
    if extra_args:
        command.extend(list(extra_args))
    return command


def launch_server(
    model_path: str,
    *,
    bits: float = 3.5,
    mode: str = "safe",
    host: str = "127.0.0.1",
    port: int = 8081,
    ctx_size: int | None = None,
    installation: LlamaCppInstallation | None = None,
    allow_fallback: bool = True,
    fallback_cache_type: str = DEFAULT_FALLBACK_CACHE_TYPE,
    extra_args: Iterable[str] | None = None,
) -> subprocess.Popen[str]:
    command = build_server_command(
        model_path,
        bits=bits,
        mode=mode,
        host=host,
        port=port,
        ctx_size=ctx_size,
        installation=installation,
        allow_fallback=allow_fallback,
        fallback_cache_type=fallback_cache_type,
        extra_args=extra_args,
    )
    return subprocess.Popen(command, text=True)


def enable(
    model_path: str,
    *,
    bits: float = 3.5,
    mode: str = "safe",
    installation: LlamaCppInstallation | None = None,
    allow_fallback: bool = True,
    fallback_cache_type: str = DEFAULT_FALLBACK_CACHE_TYPE,
) -> EngineRuntime:
    installation = installation or discover_installation()
    requested_cache_type = normalize_cache_type(bits)
    resolved_cache_type, fallback_applied = resolve_cache_type(
        bits,
        installation,
        allow_fallback=allow_fallback,
        fallback_cache_type=fallback_cache_type,
    )
    return EngineRuntime(
        name="llamacpp",
        bits=bits,
        mode=mode,
        options={
            "model_path": model_path,
            "requested_cache_type": requested_cache_type,
            "resolved_cache_type": resolved_cache_type,
            "fallback_applied": fallback_applied,
            "supports_turbo_cache": installation.supports_turbo_cache,
            "server_command": build_server_command(
                model_path,
                bits=bits,
                mode=mode,
                installation=installation,
                allow_fallback=allow_fallback,
                fallback_cache_type=fallback_cache_type,
            ),
        },
    )
