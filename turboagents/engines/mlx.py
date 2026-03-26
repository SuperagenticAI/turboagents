"""MLX / MLX-LM adapter helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import subprocess

from turboagents.engines.base import AdapterStatus, EngineRuntime


@dataclass(frozen=True, slots=True)
class MLXInstallation:
    mlx_lm_init: str | None
    generate_py: str | None
    utils_py: str | None
    server_py: str | None
    supports_native_kv_bits: bool

    @property
    def available(self) -> bool:
        return self.mlx_lm_init is not None


def _find_mlx_lm_init() -> str | None:
    spec = importlib.util.find_spec("mlx_lm")
    return spec.origin if spec else None


def discover_installation() -> MLXInstallation:
    mlx_lm_init = _find_mlx_lm_init()
    if not mlx_lm_init:
        return MLXInstallation(
            mlx_lm_init=None,
            generate_py=None,
            utils_py=None,
            server_py=None,
            supports_native_kv_bits=False,
        )
    root = Path(mlx_lm_init).resolve().parent
    generate_py = root / "generate.py"
    utils_py = root / "utils.py"
    server_py = root / "server.py"
    supports_native_kv_bits = False
    if generate_py.exists():
        text = generate_py.read_text(encoding="utf-8")
        supports_native_kv_bits = "kv_bits" in text and "maybe_quantize_kv_cache" in text
    return MLXInstallation(
        mlx_lm_init=str(mlx_lm_init),
        generate_py=str(generate_py),
        utils_py=str(utils_py),
        server_py=str(server_py) if server_py.exists() else None,
        supports_native_kv_bits=supports_native_kv_bits,
    )


def status() -> AdapterStatus:
    installation = discover_installation()
    if not installation.available:
        return AdapterStatus(
            name="mlx",
            available=True,
            detail="baseline helper only; mlx_lm not installed",
        )
    detail = (
        f"mlx_lm source detected; native_kv_bits={'yes' if installation.supports_native_kv_bits else 'no'}; "
        f"http_server={'yes' if installation.server_py else 'no'}"
    )
    return AdapterStatus(name="mlx", available=True, detail=detail)


def resolve_native_kv_bits(bits: float) -> int:
    """Map requested TurboQuant bits to MLX-LM's integer kv_bits setting."""
    if bits <= 2.5:
        return 2
    if bits <= 3.5:
        return 4
    return 4


def build_generate_options(
    *,
    bits: float = 3.5,
    mode: str = "safe",
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> dict[str, int | str]:
    return {
        "mode": mode,
        "kv_bits": resolve_native_kv_bits(bits),
        "kv_group_size": kv_group_size,
        "quantized_kv_start": quantized_kv_start,
    }


def build_runtime_generate_kwargs(
    *,
    bits: float = 3.5,
    mode: str = "safe",
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> dict[str, int]:
    options = build_generate_options(
        bits=bits,
        mode=mode,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )
    options.pop("mode", None)
    return options


def _extract_sampling_options(kwargs: dict) -> tuple[dict, dict]:
    sampling = {}
    for key in (
        "temp",
        "top_p",
        "min_p",
        "min_tokens_to_keep",
        "top_k",
        "xtc_probability",
        "xtc_threshold",
        "xtc_special_tokens",
        "repetition_penalty",
        "repetition_context_size",
        "logit_bias",
    ):
        if key in kwargs:
            sampling[key] = kwargs.pop(key)
    return kwargs, sampling


def _apply_sampling_kwargs(mlx_lm_module, options: dict, sampling: dict) -> dict:
    temp = sampling.pop("temp", None)
    top_p = sampling.pop("top_p", None)
    min_p = sampling.pop("min_p", None)
    min_tokens_to_keep = sampling.pop("min_tokens_to_keep", None)
    top_k = sampling.pop("top_k", None)
    xtc_probability = sampling.pop("xtc_probability", None)
    xtc_threshold = sampling.pop("xtc_threshold", None)
    xtc_special_tokens = sampling.pop("xtc_special_tokens", None)
    repetition_penalty = sampling.pop("repetition_penalty", None)
    repetition_context_size = sampling.pop("repetition_context_size", None)
    logit_bias = sampling.pop("logit_bias", None)

    if any(
        value is not None
        for value in (
            temp,
            top_p,
            min_p,
            min_tokens_to_keep,
            top_k,
            xtc_probability,
            xtc_threshold,
            xtc_special_tokens,
        )
    ):
        sample_utils = importlib.import_module("mlx_lm.sample_utils")
        options["sampler"] = sample_utils.make_sampler(
            temp=0.0 if temp is None else temp,
            top_p=0.0 if top_p is None else top_p,
            min_p=0.0 if min_p is None else min_p,
            min_tokens_to_keep=1 if min_tokens_to_keep is None else min_tokens_to_keep,
            top_k=0 if top_k is None else top_k,
            xtc_probability=0.0 if xtc_probability is None else xtc_probability,
            xtc_threshold=0.0 if xtc_threshold is None else xtc_threshold,
            xtc_special_tokens=[] if xtc_special_tokens is None else xtc_special_tokens,
        )

    if any(value is not None for value in (repetition_penalty, repetition_context_size, logit_bias)):
        sample_utils = importlib.import_module("mlx_lm.sample_utils")
        options["logits_processors"] = sample_utils.make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=20 if repetition_context_size is None else repetition_context_size,
        )
    return options


def build_server_command(
    model_path: str,
    *,
    bits: float = 3.5,
    host: str = "127.0.0.1",
    port: int = 8080,
    adapter_path: str | None = None,
    draft_model: str | None = None,
    trust_remote_code: bool = True,
    max_tokens: int = 512,
    installation: MLXInstallation | None = None,
) -> list[str]:
    installation = installation or discover_installation()
    if not installation.available:
        raise RuntimeError("mlx_lm is not installed.")
    command = [
        "python3",
        "-m",
        "mlx_lm.server",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--max-tokens",
        str(max_tokens),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    if adapter_path:
        command.extend(["--adapter-path", adapter_path])
    if draft_model:
        command.extend(["--draft-model", draft_model])
    return command


def launch_server(
    model_path: str,
    *,
    bits: float = 3.5,
    host: str = "127.0.0.1",
    port: int = 8080,
    adapter_path: str | None = None,
    draft_model: str | None = None,
    trust_remote_code: bool = True,
    max_tokens: int = 512,
    installation: MLXInstallation | None = None,
) -> subprocess.Popen[str]:
    command = build_server_command(
        model_path,
        bits=bits,
        host=host,
        port=port,
        adapter_path=adapter_path,
        draft_model=draft_model,
        trust_remote_code=trust_remote_code,
        max_tokens=max_tokens,
        installation=installation,
    )
    return subprocess.Popen(command, text=True)


def load(
    model_path: str,
    *,
    tokenizer_config: dict | None = None,
    model_config: dict | None = None,
    adapter_path: str | None = None,
    lazy: bool = False,
):
    """Lazy import and delegate to mlx_lm.load."""
    mlx_lm = importlib.import_module("mlx_lm")
    return mlx_lm.load(
        model_path,
        tokenizer_config=tokenizer_config or {},
        model_config=model_config or {},
        adapter_path=adapter_path,
        lazy=lazy,
    )


def enable(
    model: object,
    *,
    bits: float = 3.5,
    mode: str = "safe",
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> EngineRuntime:
    installation = discover_installation()
    runtime = EngineRuntime(
        name="mlx",
        bits=bits,
        mode=mode,
        options={
            "supports_native_kv_bits": installation.supports_native_kv_bits,
            "native_options": build_generate_options(
                bits=bits,
                mode=mode,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
            ),
        },
    )
    try:
        setattr(model, "_turboagents_runtime", runtime)
    except Exception:
        pass
    return runtime


def generate(
    model: object,
    tokenizer: object,
    prompt: str | list[int],
    *,
    bits: float = 3.5,
    mode: str = "safe",
    **kwargs,
):
    """Lazy import and delegate to mlx_lm.generate with native kv_bits wiring."""
    mlx_lm = importlib.import_module("mlx_lm")
    options = build_runtime_generate_kwargs(bits=bits, mode=mode)
    kwargs, sampling = _extract_sampling_options(dict(kwargs))
    options.update(kwargs)
    options = _apply_sampling_kwargs(mlx_lm, options, sampling)
    return mlx_lm.generate(model, tokenizer, prompt, **options)


def stream_generate(
    model: object,
    tokenizer: object,
    prompt: str | list[int],
    *,
    bits: float = 3.5,
    mode: str = "safe",
    **kwargs,
):
    """Lazy import and delegate to mlx_lm.stream_generate."""
    mlx_lm = importlib.import_module("mlx_lm")
    options = build_runtime_generate_kwargs(bits=bits, mode=mode)
    kwargs, sampling = _extract_sampling_options(dict(kwargs))
    options.update(kwargs)
    options = _apply_sampling_kwargs(mlx_lm, options, sampling)
    return mlx_lm.stream_generate(model, tokenizer, prompt, **options)


def enable_server(
    model_path: str,
    *,
    bits: float = 3.5,
    mode: str = "safe",
    host: str = "127.0.0.1",
    port: int = 8080,
    adapter_path: str | None = None,
    draft_model: str | None = None,
    installation: MLXInstallation | None = None,
) -> EngineRuntime:
    installation = installation or discover_installation()
    native_options = build_generate_options(bits=bits, mode=mode)
    return EngineRuntime(
        name="mlx",
        bits=bits,
        mode=mode,
        options={
            "supports_native_kv_bits": installation.supports_native_kv_bits,
            "native_options": native_options,
            "server_command": build_server_command(
                model_path,
                bits=bits,
                host=host,
                port=port,
                adapter_path=adapter_path,
                draft_model=draft_model,
                installation=installation,
            ),
        },
    )
