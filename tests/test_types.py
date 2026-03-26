from turboagents.quant import Config, quantize
from turboagents.quant.types import CompressedVector


def test_compressed_vector_roundtrips_through_dict() -> None:
    cfg = Config(bits=3.5, head_dim=64, seed=5)
    vector = [float(i) / 64.0 for i in range(64)]
    payload = quantize(vector, cfg)
    restored = CompressedVector.from_dict(payload.to_dict())

    assert restored.seed == payload.seed
    assert restored.bits == payload.bits
    assert restored.angle_indices.shape == payload.angle_indices.shape
    assert restored.residual_signs.shape == payload.residual_signs.shape
    assert restored.residual_group_norms.shape == payload.residual_group_norms.shape


def test_compressed_vector_roundtrips_through_bytes() -> None:
    cfg = Config(bits=3.0, head_dim=64, seed=9)
    vector = [float(i) / 10.0 for i in range(64)]
    payload = quantize(vector, cfg)
    restored = CompressedVector.from_bytes(payload.to_bytes())

    assert restored.seed == payload.seed
    assert restored.mode == payload.mode
    assert restored.bits == payload.bits
    assert restored.angle_indices.tolist() == payload.angle_indices.tolist()
    assert restored.residual_signs.tolist() == payload.residual_signs.tolist()
    assert restored.residual_group_norms.tolist() == payload.residual_group_norms.tolist()
