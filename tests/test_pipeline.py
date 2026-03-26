import numpy as np

import turboagents as ta


def test_quantize_dequantize_roundtrip_placeholder() -> None:
    cfg = ta.Config(bits=3.5, head_dim=64, seed=3)
    vector = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    compressed = ta.quantize(vector, cfg)
    restored = ta.dequantize(compressed, cfg)
    assert restored.shape == vector.shape
    assert np.linalg.norm(vector - restored) < 2.0


def test_quantize_returns_structured_payload() -> None:
    cfg = ta.Config(bits=3.0, head_dim=64, seed=1)
    vector = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
    compressed = ta.quantize(vector, cfg)
    assert compressed.angle_indices.shape == (63,)
    assert compressed.residual_signs.shape == (64,)
    assert compressed.residual_group_norms.shape == (0,)
    assert compressed.radius >= 0.0
    assert compressed.residual_norm >= 0.0


def test_inner_product_runs_in_compressed_domain() -> None:
    cfg = ta.Config(bits=3.5, head_dim=64, seed=2)
    a = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    b = np.linspace(1.0, -1.0, 64, dtype=np.float32)
    compressed = ta.quantize(a, cfg)
    score = ta.inner_product(b, compressed, cfg)
    assert isinstance(score, float)
    restored = ta.dequantize(compressed, cfg)
    expected = float(np.dot(b, restored))
    assert abs(score - expected) < 1e-4
