import numpy as np

from turboagents.quant.config import Config
from turboagents.quant.hadamard import fwht, inverse_rotate, rotate, sign_pattern


def test_fwht_preserves_norm() -> None:
    vector = np.arange(8, dtype=np.float32)
    transformed = fwht(vector)
    assert np.isclose(np.linalg.norm(vector), np.linalg.norm(transformed), atol=1e-5)


def test_rotate_then_inverse_rotate_restores_vector() -> None:
    cfg = Config(bits=3.5, head_dim=8 * 8, seed=7)
    vector = np.linspace(-1.0, 1.0, cfg.head_dim, dtype=np.float32)
    rotated = rotate(vector, cfg)
    restored = inverse_rotate(rotated, cfg)
    assert np.allclose(vector, restored, atol=1e-5)


def test_sign_pattern_is_deterministic_and_read_only() -> None:
    a = sign_pattern(64, 7)
    b = sign_pattern(64, 7)
    assert np.array_equal(a, b)
    assert a.flags.writeable is False
