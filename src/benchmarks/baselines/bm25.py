"""BM25 baseline — Naive RAG per paper Table 2.

Tokenises the ingested messages with whitespace + lower-casing, builds an
in-memory ``BM25Okapi`` index, retrieves the top-k passages by BM25 score
for each query, and answers with the same Azure chat deployment used by
the dense baselines. No memory lifecycle, no decay, no pyramid — this is
the sparse-retrieval reference system.
"""

from __future__ import annotations

from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from benchmarks.baselines.base import Baseline


_SYSTEM = "You answer questions using ONLY the provided memory context. Be concise."


class BM25Baseline(Baseline):
    """Sparse BM25 retrieval + LLM answerer. Paper's Naive RAG reference."""

    name = "bm25"

    def __init__(self, azure_client, top_k: int = 5, deployment: str = "gpt-4.1") -> None:
        self.azure_client = azure_client
        self.top_k = top_k
        self.deployment = deployment
        self.corpus: List[str] = []
        self._bm25 = None
        self.last_retrieved: list = []

    def reset(self) -> None:
        self.corpus = []
        self._bm25 = None
        self.last_retrieved = []

    def ingest(self, messages: list) -> None:
        for msg in messages:
            txt = msg if isinstance(msg, str) else getattr(msg, "content", str(msg))
            self.corpus.append(txt)
        tokenized = [c.lower().split() for c in self.corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def query(self, question: str) -> str:
        if not self._bm25:
            self.last_retrieved = []
            return "I do not have enough information to answer."
        tokens = question.lower().split()
        scores = self._bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][: self.top_k]
        ctx = [self.corpus[i] for i in idxs if scores[i] > 0]
        self.last_retrieved = list(ctx)
        if not ctx:
            return "I do not have enough information to answer."
        prompt = (
            "Context:\n"
            + "\n---\n".join(ctx)
            + f"\n\nQuestion: {question}\nAnswer:"
        )
        # AzureClient.complete signature is (prompt, system, ...) — keep the
        # call shape consistent with the dense baselines so the answerer
        # cache key shape matches when prompts are identical.
        return self.azure_client.complete(
            prompt=prompt, system=_SYSTEM, category="completion",
            temperature=0, max_tokens=300,
        )

    def cost_so_far(self) -> dict:
        if hasattr(self.azure_client, "tracker"):
            t = self.azure_client.tracker
            return {
                "embedding": t.embedding,
                "completion": t.completion,
                "total": t.total_tokens(),
            }
        return super().cost_so_far()
