"""Core types used by the quantization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

import numpy as np

from turboagents.quant.codebooks import level_count


Array = np.ndarray
MODE_TO_ID = {"mse": 0, "prod": 1}
ID_TO_MODE = {value: key for key, value in MODE_TO_ID.items()}


def _pack_uint_values(values: np.ndarray, bits_per_value: int) -> bytes:
    if bits_per_value <= 0:
        raise ValueError("bits_per_value must be positive.")
    packed = 0
    bit_count = 0
    out = bytearray()
    mask = (1 << bits_per_value) - 1
    for raw in values.astype(np.int64):
        value = int(raw)
        if value < 0 or value > mask:
            raise ValueError(f"Value {value} does not fit in {bits_per_value} bits.")
        packed |= value << bit_count
        bit_count += bits_per_value
        while bit_count >= 8:
            out.append(packed & 0xFF)
            packed >>= 8
            bit_count -= 8
    if bit_count:
        out.append(packed & 0xFF)
    return bytes(out)


def _unpack_uint_values(payload: bytes, count: int, bits_per_value: int) -> np.ndarray:
    if bits_per_value <= 0:
        raise ValueError("bits_per_value must be positive.")
    values = np.zeros(count, dtype=np.int32)
    data = memoryview(payload)
    acc = 0
    bit_count = 0
    byte_idx = 0
    mask = (1 << bits_per_value) - 1
    for idx in range(count):
        while bit_count < bits_per_value:
            if byte_idx >= len(data):
                raise ValueError("Packed payload ended before all values were decoded.")
            acc |= int(data[byte_idx]) << bit_count
            bit_count += 8
            byte_idx += 1
        values[idx] = acc & mask
        acc >>= bits_per_value
        bit_count -= bits_per_value
    return values


def _pack_signs(signs: np.ndarray) -> bytes:
    bits = np.where(signs >= 0, 1, 0).astype(np.uint8)
    return np.packbits(bits, bitorder="little").tobytes()


def _unpack_signs(payload: bytes, count: int) -> np.ndarray:
    unpacked = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        bitorder="little",
    )[:count]
    return np.where(unpacked == 1, 1, -1).astype(np.int8)


@dataclass(slots=True)
class CompressedVector:
    """Structured compressed representation.

    This is not yet the final bit-packed wire format, but it now mirrors the
    intended two-stage shape:

    - angle indices for the PolarQuant-style stage
    - sign bits for the residual correction stage
    - radius and residual norm scalars
    """

    angle_indices: Array
    residual_signs: Array
    residual_group_norms: Array
    radius: float
    residual_norm: float
    bits: float
    seed: int
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_indices": self.angle_indices.astype(np.int32).tolist(),
            "residual_signs": self.residual_signs.astype(np.int8).tolist(),
            "residual_group_norms": self.residual_group_norms.astype(np.float32).tolist(),
            "radius": float(self.radius),
            "residual_norm": float(self.residual_norm),
            "bits": float(self.bits),
            "seed": int(self.seed),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompressedVector":
        return cls(
            angle_indices=np.asarray(payload["angle_indices"], dtype=np.int32),
            residual_signs=np.asarray(payload["residual_signs"], dtype=np.int8),
            residual_group_norms=np.asarray(payload["residual_group_norms"], dtype=np.float32),
            radius=float(payload["radius"]),
            residual_norm=float(payload["residual_norm"]),
            bits=float(payload["bits"]),
            seed=int(payload["seed"]),
            mode=str(payload["mode"]),
        )

    @property
    def angle_bits_per_index(self) -> int:
        levels = level_count(self.bits)
        return int(np.ceil(np.log2(levels)))

    @property
    def estimated_size_bytes(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        angle_payload = _pack_uint_values(self.angle_indices, self.angle_bits_per_index)
        sign_payload = _pack_signs(self.residual_signs)
        header = struct.pack(
            "<4sBfHqBfI",
            b"TQCV",
            1,
            float(self.bits),
            int(self.angle_indices.size),
            int(self.seed),
            int(MODE_TO_ID[self.mode]),
            float(self.radius),
            int(self.residual_signs.size),
        )
        group_norms = self.residual_group_norms.astype(np.float32).tobytes()
        residual_header = struct.pack(
            "<fIII",
            float(self.residual_norm),
            len(angle_payload),
            len(sign_payload),
            len(group_norms),
        )
        return header + residual_header + angle_payload + sign_payload + group_norms

    @classmethod
    def from_bytes(cls, payload: bytes) -> "CompressedVector":
        header_size = struct.calcsize("<4sBfHqBfI")
        residual_header_size = struct.calcsize("<fIII")
        if len(payload) < header_size + residual_header_size:
            raise ValueError("Compressed payload is too short.")
        magic, version, bits, angle_count, seed, mode_id, radius, sign_count = struct.unpack(
            "<4sBfHqBfI", payload[:header_size]
        )
        if magic != b"TQCV":
            raise ValueError("Invalid compressed payload magic.")
        if version != 1:
            raise ValueError(f"Unsupported compressed payload version: {version}")
        residual_norm, angle_len, sign_len, group_norm_len = struct.unpack(
            "<fIII", payload[header_size : header_size + residual_header_size]
        )
        body = payload[header_size + residual_header_size :]
        if len(body) != angle_len + sign_len + group_norm_len:
            raise ValueError("Compressed payload body length mismatch.")
        angle_payload = body[:angle_len]
        sign_payload = body[angle_len : angle_len + sign_len]
        group_norm_payload = body[angle_len + sign_len :]
        angle_bits = int(np.ceil(np.log2(level_count(bits))))
        angle_indices = _unpack_uint_values(angle_payload, angle_count, angle_bits)
        residual_signs = _unpack_signs(sign_payload, sign_count)
        residual_group_norms = np.frombuffer(group_norm_payload, dtype=np.float32).copy()
        return cls(
            angle_indices=angle_indices,
            residual_signs=residual_signs,
            residual_group_norms=residual_group_norms,
            radius=float(radius),
            residual_norm=float(residual_norm),
            bits=float(bits),
            seed=int(seed),
            mode=ID_TO_MODE[int(mode_id)],
        )
