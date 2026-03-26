from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import turboagents as ta


cfg = ta.Config(bits=3.5, head_dim=64, seed=42)
vector = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
compressed = ta.quantize(vector, cfg)
packed = compressed.to_bytes()
unpacked = type(compressed).from_bytes(packed)
restored = ta.dequantize(compressed, cfg)

print("compression_ratio_vs_fp16:", cfg.compression_ratio_vs_fp16)
print("serialized_bytes:", len(packed))
print("angle_indices:", unpacked.angle_indices.shape[0])
print("restored_l2_error:", float(np.linalg.norm(vector - restored)))
