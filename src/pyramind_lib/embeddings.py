"""Embedding backends: BGE (paper default) and Azure (cost-friendly alternative).

The paper specifies BAAI/bge-large-en-v1.5 (1024-d, fp32) as the canonical
embedding model. The Azure backend (text-embedding-3-large, 3072-d) is
documented as a drop-in cost/latency alternative — it is the model used by
the existing AzureClient in ``pyramind_lib.azure`` and what the benchmark
sweeps default to today.

These backends are surfaced as an explicit abstraction so paper-faithful
runs can switch to BGE without touching the engine, the cache layout, or the
orchestrator wiring. The engine itself currently embeds via ``AzureClient``;
re-wiring it to consume an ``Embedder`` is a future change deliberately kept
out of this patch to avoid breaking existing run artifacts.
"""
from __future__ import annotations

from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    """Minimal embedding-backend protocol consumed by the engine.

    Implementations must expose the output dimensionality so callers can
    pre-allocate / sanity-check vectors, and ``embed`` must return an
    ``(N, dim)`` float32 array.
    """

    dim: int

    def embed(self, texts: List[str]) -> np.ndarray: ...


class BGEEmbedder:
    """sentence-transformers BAAI/bge-large-en-v1.5 (1024-d fp32). Paper default."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        # Lazy import so the heavy torch / sentence-transformers stack is only
        # required when this backend is actually selected.
        from sentence_transformers import SentenceTransformer  # noqa: WPS433
        self._model = SentenceTransformer(model_name)
        self.dim = 1024

    def embed(self, texts: List[str]) -> np.ndarray:
        emb = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
        )
        return emb.astype(np.float32)


class AzureEmbedder:
    """Azure text-embedding-3-large (3072-d). Cost/latency alternative.

    Adapts the ``AzureClient.embed(text, model=...)`` signature used by the
    existing client. The deployment kwarg is forwarded as ``model=`` since
    that is what the client exposes.
    """

    def __init__(self, azure_client, deployment: str = "text-embedding-3-large") -> None:
        self._client = azure_client
        self._deployment = deployment
        self.dim = 3072

    def embed(self, texts: List[str]) -> np.ndarray:
        out = []
        for t in texts:
            v = self._client.embed(t, model=self._deployment)
            out.append(np.asarray(v, dtype=np.float32))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)


def make_embedder(backend: str, azure_client=None, **kwargs):
    """Construct the requested embedding backend.

    ``backend`` ∈ {"bge", "azure"}. The Azure backend additionally requires
    an ``azure_client`` to wrap.
    """
    if backend == "bge":
        return BGEEmbedder(**kwargs)
    if backend == "azure":
        if azure_client is None:
            raise ValueError("azure backend requires azure_client")
        return AzureEmbedder(azure_client, **kwargs)
    raise ValueError(f"Unknown embedding backend: {backend}")
