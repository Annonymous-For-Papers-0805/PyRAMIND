"""Dense RAG baseline — embed every message, retrieve top-k by cosine, generate answer.

This is the DENSE variant of Naive RAG (Azure ``text-embedding-3-large`` +
cosine similarity). The sparse counterpart lives in ``bm25.py`` and is the
"Naive RAG" row in paper Table 2; we keep both so the ablation can attribute
score deltas to the retriever choice rather than to RAG-vs-memory.

The class is registered under two system names in the orchestrator:
``"naive-rag"`` (legacy / backward-compat) and ``"dense-rag"`` (preferred).

No decay. No dedup. No pyramid. The "do you really need a memory
architecture?" sanity baseline.
"""

from __future__ import annotations

from benchmarks.baselines.base import Baseline
from pyramind_lib.retrieval import cosine_similarity, top_k_indices

_SYSTEM = "You answer questions using ONLY the provided memory context. Be concise."


class NaiveRAGBaseline(Baseline):
    """Dense RAG (Azure embedding + cosine top-k). No memory lifecycle.

    Kept under the historical class name for ABI stability; the orchestrator
    accepts both ``--system naive-rag`` (legacy alias) and
    ``--system dense-rag`` (preferred) and both route here.
    """

    name = "naive-rag"

    def __init__(self, azure_client, deployment: str = "gpt-4.1", top_k: int = 5) -> None:
        self.azure_client = azure_client
        self.deployment = deployment
        self.top_k = top_k
        self.chunks: list = []
        self.embeddings: list = []

    def ingest(self, messages: list) -> None:
        for msg in messages:
            emb = self.azure_client.embed(msg)
            self.chunks.append(msg)
            self.embeddings.append(emb)

    def query(self, question: str) -> str:
        if not self.chunks:
            return "I do not have enough information to answer."

        q_emb = self.azure_client.embed(question)
        scores = [cosine_similarity(q_emb, e) for e in self.embeddings]
        top_idx = top_k_indices(scores, k=self.top_k)
        retrieved = [self.chunks[i] for i in top_idx]

        context = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(retrieved))
        prompt = (
            f"Memory context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer concisely using only the context."
        )
        return self.azure_client.complete(
            prompt=prompt, system=_SYSTEM, category="completion",
            temperature=0, max_tokens=300,
        )

    def reset(self) -> None:
        self.chunks = []
        self.embeddings = []

    def cost_so_far(self) -> dict:
        if hasattr(self.azure_client, "tracker"):
            t = self.azure_client.tracker
            return {
                "embedding": t.embedding,
                "completion": t.completion,
                "total": t.total_tokens(),
            }
        return super().cost_so_far()
