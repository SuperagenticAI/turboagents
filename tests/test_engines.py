from turboagents.engines.llamacpp import (
    LlamaCppInstallation,
    build_server_command,
    enable as enable_llamacpp,
    resolve_cache_type,
)
from turboagents.engines.mlx import (
    MLXInstallation,
    build_generate_options,
    build_runtime_generate_kwargs,
    build_server_command as build_mlx_server_command,
    enable as enable_mlx,
    enable_server as enable_mlx_server,
    resolve_native_kv_bits,
)
from turboagents.engines.vllm import (
    VllmInstallation,
    build_serve_command as build_vllm_serve_command,
    enable as enable_vllm,
    resolve_kv_cache_dtype,
)


def test_llamacpp_enable_returns_runtime() -> None:
    installation = LlamaCppInstallation(
        server_executable="/usr/local/bin/llama-server",
        cli_executable="/usr/local/bin/llama-cli",
        help_text="",
        supported_cache_types=("f16", "turbo3.5"),
    )
    runtime = enable_llamacpp("model.gguf", bits=3.5, installation=installation)
    assert runtime.name == "llamacpp"
    assert runtime.options["requested_cache_type"] == "turbo3.5"
    assert runtime.options["resolved_cache_type"] == "turbo3.5"


def test_llamacpp_falls_back_when_turbo_not_supported() -> None:
    installation = LlamaCppInstallation(
        server_executable="/usr/local/bin/llama-server",
        cli_executable="/usr/local/bin/llama-cli",
        help_text="allowed values: f32, f16, q4_0",
        supported_cache_types=("f16", "f32", "q4_0"),
    )
    cache_type, fallback_applied = resolve_cache_type(3.5, installation, allow_fallback=True)
    assert cache_type == "f16"
    assert fallback_applied is True


def test_llamacpp_build_server_command_uses_resolved_cache_type() -> None:
    installation = LlamaCppInstallation(
        server_executable="/usr/local/bin/llama-server",
        cli_executable="/usr/local/bin/llama-cli",
        help_text="allowed values: f32, f16, turbo3.5",
        supported_cache_types=("f16", "f32", "turbo3.5"),
    )
    command = build_server_command(
        "model.gguf",
        bits=3.5,
        installation=installation,
        host="0.0.0.0",
        port=9000,
    )
    assert command[0] == "/usr/local/bin/llama-server"
    assert "--cache-type-k" in command
    assert "turbo3.5" in command
    assert "9000" in command


def test_mlx_enable_attaches_runtime() -> None:
    class Dummy:
        pass

    model = Dummy()
    runtime = enable_mlx(model, bits=3.0)
    assert runtime.name == "mlx"
    assert getattr(model, "_turboagents_runtime").bits == 3.0
    assert runtime.options["native_options"]["kv_bits"] == 4


def test_mlx_resolves_native_kv_bits() -> None:
    assert resolve_native_kv_bits(2.0) == 2
    assert resolve_native_kv_bits(3.0) == 4
    assert build_generate_options(bits=3.5)["kv_bits"] == 4
    assert "mode" not in build_runtime_generate_kwargs(bits=3.5)


def test_mlx_server_runtime_builds_command() -> None:
    installation = MLXInstallation(
        mlx_lm_init="/opt/miniconda3/lib/python3.13/site-packages/mlx_lm/__init__.py",
        generate_py="/opt/miniconda3/lib/python3.13/site-packages/mlx_lm/generate.py",
        utils_py="/opt/miniconda3/lib/python3.13/site-packages/mlx_lm/utils.py",
        server_py="/opt/miniconda3/lib/python3.13/site-packages/mlx_lm/server.py",
        supports_native_kv_bits=True,
    )
    runtime = enable_mlx_server(
        "mlx-community/Qwen3-0.6B-4bit",
        bits=3.5,
        host="0.0.0.0",
        port=9000,
        installation=installation,
    )
    assert runtime.name == "mlx"
    assert "--model" in runtime.options["server_command"]
    assert "9000" in runtime.options["server_command"]


def test_mlx_build_server_command_uses_python_module() -> None:
    installation = MLXInstallation(
        mlx_lm_init="/opt/miniconda3/lib/python3.13/site-packages/mlx_lm/__init__.py",
        generate_py="/opt/miniconda3/lib/python3.13/site-packages/mlx_lm/generate.py",
        utils_py="/opt/miniconda3/lib/python3.13/site-packages/mlx_lm/utils.py",
        server_py="/opt/miniconda3/lib/python3.13/site-packages/mlx_lm/server.py",
        supports_native_kv_bits=True,
    )
    command = build_mlx_server_command(
        "mlx-community/Qwen3-0.6B-4bit",
        bits=3.5,
        host="0.0.0.0",
        port=9000,
        installation=installation,
    )
    assert command[:3] == ["python3", "-m", "mlx_lm.server"]
    assert "--model" in command


def test_vllm_config_is_marked_experimental() -> None:
    installation = VllmInstallation(
        executable="/usr/local/bin/vllm",
        version="0.8.2",
        import_available=True,
        plugin_entrypoint="turboagents",
        help_text="--kv-cache-dtype",
        supported_kv_cache_dtypes=("auto", "fp8", "fp8_e4m3"),
    )
    runtime = enable_vllm("meta-llama/Llama-3.1-8B-Instruct", bits=3.5, installation=installation)
    assert runtime.name == "vllm"
    assert runtime.options["resolved_kv_cache_dtype"] == "fp8"
    assert runtime.options["env"]["VLLM_PLUGINS"] == "turboagents"


def test_vllm_resolves_to_fp8_by_default() -> None:
    dtype, fallback = resolve_kv_cache_dtype(3.5)
    assert dtype == "fp8"
    assert fallback is True


def test_vllm_build_server_command_uses_kv_cache_dtype() -> None:
    installation = VllmInstallation(
        executable="/usr/local/bin/vllm",
        version="0.8.2",
        import_available=True,
        plugin_entrypoint="turboagents",
        help_text="--kv-cache-dtype",
        supported_kv_cache_dtypes=("auto", "fp8", "fp8_e4m3"),
    )
    command, env = build_vllm_serve_command(
        "meta-llama/Llama-3.1-8B-Instruct",
        bits=3.5,
        installation=installation,
        host="0.0.0.0",
        port=9000,
    )
    assert command[0] == "/usr/local/bin/vllm"
    assert command[1] == "serve"
    assert "--kv-cache-dtype" in command
    assert "fp8" in command
    assert env["VLLM_PLUGINS"] == "turboagents"
