"""Deterministic R scoring (Innovation B).

Paper Eq. 2: R = clip(cos(e_m, c) mapped from [-1, 1] to [R_min, R_max]).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pyramind_lib.config import PyramindConfig
from pyramind_lib.retrieval import cosine_similarity


def compute_R(
    content: str,
    embedding,
    context_vector: Optional[np.ndarray],
    config: PyramindConfig,
) -> float:
    """Compute the R score for a memory.

    Innovation B: when `enable_deterministic_R=True` AND a `context_vector` is
    available, R is the cosine alignment between the memory's embedding and the
    context vector, mapped from [-1, 1] to [config.R_min, config.R_max].
    Otherwise returns `config.default_R`.
    """
    if config.enable_deterministic_R and context_vector is not None:
        sim = cosine_similarity(embedding, context_vector.tolist())
        # map [-1, 1] -> [R_min, R_max] (paper Eq. 2)
        return max(config.R_min, min(config.R_max, (sim + 1.0) / 2.0))
    return config.default_R
