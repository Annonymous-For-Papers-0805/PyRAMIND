"""PyRAMIND baseline — wraps MemoryEngine in the Baseline interface."""

from __future__ import annotations

from benchmarks.baselines.base import Baseline
from pyramind_lib.config import PyramindConfig
from pyramind_lib.engine import MemoryEngine

_SYSTEM = "You answer questions using ONLY the provided memory context. Be concise."


class PyramindBaseline(Baseline):
    """Wraps MemoryEngine. Each ingested message triggers record() + run_cycle()."""

    name = "pyramind"

    def __init__(
        self,
        azure_client,
        config: PyramindConfig,
        formula_name: str = "ensemble",
        top_k: int = 5,
        deployment: str = "gpt-4.1",
    ) -> None:
        self.azure_client = azure_client
        self.config = config
        self.formula_name = formula_name
        self.top_k = top_k
        self.deployment = deployment
        self.engine = MemoryEngine(config, azure_client, formula_name=formula_name)
        self.last_retrieved: list = []  # populated by query() — for recall logging

    def ingest(self, messages: list) -> None:
        for msg in messages:
            self.engine.record(msg)
            self.engine.run_cycle()

    def query(self, question: str) -> str:
        result = self.engine.query(question, top_k=self.top_k)
        # Stash retrieved context so the runner can compute recall@k after.
        self.last_retrieved = list(result.all_context)
        if not result.all_context:
            return "I do not have enough information to answer."

        context = "\n".join(result.all_context)
        prompt = (
            f"Memory context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer concisely using only the context."
        )
        return self.azure_client.complete(
            prompt=prompt, system=_SYSTEM, category="completion",
            temperature=0, max_tokens=300,
        )

    def reset(self) -> None:
        self.engine.reset()

    def cost_so_far(self) -> dict:
        if hasattr(self.azure_client, "tracker"):
            t = self.azure_client.tracker
            return {
                "embedding": t.embedding,
                "scoring": t.scoring,
                "completion": t.completion,
                "total": t.total_tokens(),
            }
        return super().cost_so_far()

    def memory_state_hash(self) -> str:
        """Expose engine's hash for determinism testing."""
        return self.engine.memory_state_hash()
