from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turboagents.rag import TurboFAISS


rng = np.random.default_rng(0)
vectors = rng.standard_normal((64, 128), dtype=np.float32)
query = vectors[5].copy()

index = TurboFAISS(dim=128, bits=3.5, seed=0)
index.add(vectors, metadata=[{"row": idx} for idx in range(len(vectors))])
results = index.search(query, k=5, rerank_top=16)

for item in results:
    print(item)
