from turboagents.quant.config import Config


def test_config_accepts_supported_values() -> None:
    cfg = Config(bits=3.5, head_dim=128, seed=42, mode="mse")
    assert cfg.compression_ratio_vs_fp16 > 4.0


def test_config_rejects_unsupported_bits() -> None:
    try:
        Config(bits=3.2, head_dim=128)
    except ValueError as exc:
        assert "Unsupported bits" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

