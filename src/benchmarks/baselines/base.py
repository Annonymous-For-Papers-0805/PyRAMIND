"""Abstract base class for all memory baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Baseline(ABC):
    """A memory system that ingests messages then answers questions."""

    name: str = "base"

    @abstractmethod
    def ingest(self, messages: list) -> None:
        ...

    @abstractmethod
    def query(self, question: str) -> str:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    def cost_so_far(self) -> dict:
        """Return current token usage. Subclasses with native trackers should override."""
        return {"embedding": 0, "completion": 0, "total": 0}
