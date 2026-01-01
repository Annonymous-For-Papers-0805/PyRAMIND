"""Pyramidal compression (Innovation C) — k-means themes + SVD worldview.

Deterministic by virtue of `random_state=42` on KMeans and stable order of
input facts. The pyramid is a no-op when `enable_pyramid=False`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from pyramind_lib.config import PyramindConfig
from pyramind_lib.types import MemoryNode, Theme


def rebuild(
    facts: list, config: PyramindConfig, random_state: int = 42
) -> tuple:
    """Rebuild themes (level 3) and worldview (level 4) from a list of facts.

    Returns (themes, worldview_vector, worldview_summary).

    - With < 2 facts: returns empty themes, no worldview.
    - With 1 theme: worldview = theme centroid (normalized).
    - With 2+ themes: worldview = top SVD singular vector across theme centroids;
      summary = strongest theme's representative content.
    """
    if not config.enable_pyramid or len(facts) < 2:
        return [], None, ""

    n_clusters = min(config.max_theme_clusters, len(facts))
    embeddings_matrix = np.array([f.effective_embedding for f in facts])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings_matrix)

    themes: list = []
    for i in range(n_clusters):
        cluster_facts = [f for f, lbl in zip(facts, labels) if lbl == i]
        if not cluster_facts:
            continue
        representative = max(cluster_facts, key=lambda f: f.M)
        themes.append(
            Theme(
                centroid=kmeans.cluster_centers_[i],
                summary=representative.content,
                fact_ids=[f.id for f in cluster_facts],
                strength=sum(f.M for f in cluster_facts) / len(cluster_facts),
            )
        )

    if len(themes) >= 2:
        theme_matrix = np.array([t.centroid for t in themes])
        try:
            _U, _S, Vt = np.linalg.svd(theme_matrix, full_matrices=False)
            worldview = Vt[0]
            norm = np.linalg.norm(worldview)
            if norm > 0:
                worldview = worldview / norm
            strongest = max(themes, key=lambda t: t.strength)
            return themes, worldview, strongest.summary
        except np.linalg.LinAlgError:
            return themes, None, ""
    elif len(themes) == 1:
        worldview = themes[0].centroid
        norm = np.linalg.norm(worldview)
        if norm > 0:
            worldview = worldview / norm
        return themes, worldview, themes[0].summary

    return themes, None, ""
