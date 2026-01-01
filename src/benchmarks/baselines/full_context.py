"""Full-context baseline — stuff entire history into a single prompt."""

from __future__ import annotations

from benchmarks.baselines.base import Baseline

_SYSTEM = "You answer questions using ONLY the provided conversation history. Be concise."


class FullContextBaseline(Baseline):
    """Concatenates every ingested message into the prompt at query time."""

    name = "full-context"

    def __init__(self, azure_client, deployment: str = "gpt-4.1") -> None:
        self.azure_client = azure_client
        self.deployment = deployment
        self.history: list = []

    def ingest(self, messages: list) -> None:
        self.history.extend(messages)

    def query(self, question: str) -> str:
        joined = "\n".join(f"- {m}" for m in self.history)
        prompt = (
            f"Conversation history:\n{joined}\n\n"
            f"Question: {question}\n\nAnswer concisely."
        )
        return self.azure_client.complete(
            prompt=prompt, system=_SYSTEM, category="completion",
            temperature=0, max_tokens=300,
        )

    def reset(self) -> None:
        self.history = []

    def cost_so_far(self) -> dict:
        if hasattr(self.azure_client, "tracker"):
            t = self.azure_client.tracker
            return {
                "embedding": t.embedding,
                "completion": t.completion,
                "total": t.total_tokens(),
            }
        return super().cost_so_far()
