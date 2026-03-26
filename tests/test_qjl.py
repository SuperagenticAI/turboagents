import warnings

import numpy as np

from turboagents.quant.qjl import decode_residual, encode_residual, inner_product


def test_qjl_roundtrip_shape_and_nonzero_norm() -> None:
    rng = np.random.default_rng(11)
    residual = rng.standard_normal(64, dtype=np.float32)
    signs, residual_norm, group_norms = encode_residual(residual, seed=11)
    restored = decode_residual(signs, residual_norm, group_norms, seed=11)

    assert signs.shape == (64,)
    assert residual_norm > 0.0
    assert group_norms.shape == (0,)
    assert restored.shape == (64,)


def test_qjl_inner_product_matches_decoded_residual() -> None:
    rng = np.random.default_rng(12)
    residual = rng.standard_normal(64, dtype=np.float32)
    query = rng.standard_normal(64, dtype=np.float32)
    signs, residual_norm, group_norms = encode_residual(residual, seed=12)
    restored = decode_residual(signs, residual_norm, group_norms, seed=12)
    score = inner_product(query, signs, residual_norm, group_norms, seed=12)

    assert np.isclose(score, float(np.dot(query, restored)), atol=1e-4)


def test_qjl_operations_do_not_emit_runtime_warnings() -> None:
    rng = np.random.default_rng(13)
    residual = rng.standard_normal(64, dtype=np.float32)
    query = rng.standard_normal(64, dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        signs, residual_norm, group_norms = encode_residual(residual, seed=13)
        decode_residual(signs, residual_norm, group_norms, seed=13)
        inner_product(query, signs, residual_norm, group_norms, seed=13)

    runtime_warnings = [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert runtime_warnings == []
