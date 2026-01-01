"""Retrieval helpers: cosine similarity + top-k."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a, b) -> float:
    """Cosine similarity in [-1, 1]. Returns 0 if either vector has zero norm."""
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a_np))
    nb = float(np.linalg.norm(b_np))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a_np @ b_np / (na * nb))


def top_k_indices(scores, k: int) -> list:
    """Indices of top-k scores in descending order. Stable on ties."""
    if k <= 0:
        return []
    pairs = list(enumerate(scores))
    pairs.sort(key=lambda p: (-p[1], p[0]))
    return [i for i, _ in pairs[: min(k, len(pairs))]]
