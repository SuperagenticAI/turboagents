from unittest.mock import patch

from turboagents.engines.base import EngineRuntime
from turboagents.cli.serve import run


def test_serve_proxy_dry_run() -> None:
    text = run(backend="proxy", host="127.0.0.1", port=9000, dry_run=True)
    assert "127.0.0.1:9000" in text


def test_serve_llamacpp_dry_run_requires_model() -> None:
    try:
        run(backend="llamacpp", dry_run=True)
    except ValueError as exc:
        assert "--model is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_serve_vllm_dry_run_requires_model() -> None:
    try:
        run(backend="vllm", dry_run=True)
    except ValueError as exc:
        assert "--model is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_serve_mlx_dry_run_requires_model() -> None:
    try:
        run(backend="mlx", dry_run=True)
    except ValueError as exc:
        assert "--model is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_serve_vllm_dry_run_renders_env_prefixed_command() -> None:
    runtime = EngineRuntime(
        name="vllm",
        bits=3.5,
        mode="safe",
        options={
            "server_command": ["vllm", "serve", "model"],
            "env": {
                "VLLM_PLUGINS": "turboagents",
                "TURBOAGENTS_VLLM_MODE": "safe",
                "TURBOAGENTS_VLLM_BITS": "3.5",
            },
        },
    )
    with patch("turboagents.cli.serve.vllm.enable", return_value=runtime):
        text = run(backend="vllm", model="model", dry_run=True)
    assert text.startswith("VLLM_PLUGINS=turboagents")
    assert "vllm serve model" in text


def test_serve_mlx_dry_run_renders_command() -> None:
    runtime = EngineRuntime(
        name="mlx",
        bits=3.5,
        mode="safe",
        options={
            "server_command": ["python3", "-m", "mlx_lm.server", "--model", "model"],
        },
    )
    with patch("turboagents.cli.serve.mlx.enable_server", return_value=runtime):
        text = run(backend="mlx", model="model", dry_run=True)
    assert text.startswith("python3 -m mlx_lm.server")
