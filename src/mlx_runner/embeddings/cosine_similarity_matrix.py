from __future__ import annotations

import math
from typing import List, Sequence


def cosine_similarity_matrix(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    """Pairwise cosine similarity for a list of embedding vectors."""
    norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in vectors]
    out: List[List[float]] = []
    for i, vi in enumerate(vectors):
        row = []
        for j, vj in enumerate(vectors):
            dot = sum(a * b for a, b in zip(vi, vj))
            row.append(dot / (norms[i] * norms[j]))
        out.append(row)
    return out
