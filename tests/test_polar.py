import numpy as np

from turboagents.quant.codebooks import load_codebook
from turboagents.quant.config import Config
from turboagents.quant.polar import polar_dequantize, polar_quantize_rotated
from turboagents.quant.polar import from_spherical, to_spherical


def test_spherical_roundtrip_restores_vector() -> None:
    vector = np.array([0.25, -0.5, 0.75, -1.25], dtype=np.float32)
    radius, angles = to_spherical(vector)
    restored = from_spherical(radius, angles)
    assert np.allclose(vector, restored, atol=1e-5)


def test_zero_vector_maps_to_zero_angles() -> None:
    vector = np.zeros(4, dtype=np.float32)
    radius, angles = to_spherical(vector)
    assert radius == 0.0
    assert np.allclose(angles, 0.0)


def test_inner_codebook_is_sorted_and_bounded() -> None:
    codebook = load_codebook(3.5, angle_type="inner", remaining_dim=64)
    assert np.all(np.diff(codebook.centroids) >= 0.0)
    assert codebook.boundaries[0] == 0.0
    assert np.isclose(codebook.boundaries[-1], np.pi)


def test_more_bits_reduce_rotated_polar_reconstruction_error() -> None:
    rng = np.random.default_rng(21)
    rotated = rng.standard_normal(64, dtype=np.float32)
    rotated /= np.linalg.norm(rotated) + 1e-8
    cfg_low = Config(bits=2.0, head_dim=64, seed=0)
    cfg_high = Config(bits=4.0, head_dim=64, seed=0)

    low_idx, low_radius = polar_quantize_rotated(rotated, cfg_low)
    high_idx, high_radius = polar_quantize_rotated(rotated, cfg_high)
    low_restored = polar_dequantize(low_idx, low_radius, cfg_low)
    high_restored = polar_dequantize(high_idx, high_radius, cfg_high)

    low_err = float(np.linalg.norm(rotated - low_restored))
    high_err = float(np.linalg.norm(rotated - high_restored))
    assert high_err <= low_err
