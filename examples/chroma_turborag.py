from __future__ import annotations

import tempfile

import numpy as np

from turboagents.rag import TurboChroma


def main() -> None:
    workdir = tempfile.mkdtemp(prefix="turboagents-chroma-")
    index = TurboChroma(
        path=workdir,
        collection_name="demo",
        dim=64,
        bits=3.5,
        seed=7,
    )

    docs = np.eye(4, 64, dtype=np.float32)
    metadata = [
        {"title": "Alpha", "id": "doc-0"},
        {"title": "Beta", "id": "doc-1"},
        {"title": "Gamma", "id": "doc-2"},
        {"title": "Delta", "id": "doc-3"},
    ]

    index.create_collection("demo")
    index.add(docs, metadata=metadata)
    results = index.search(docs[0], k=3, rerank_top=8)

    print(f"Chroma workdir: {workdir}")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
