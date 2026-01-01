"""Core dataclasses used across the pyramind library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

CubeName = Literal["raw", "fact"]
_VALID_CUBES = ("raw", "fact")


@dataclass(frozen=True)
class FormulaInputs:
    """Inputs every memory formula consumes."""

    R: float
    f: int
    t: int
    tau: float
    D: float


@dataclass
class MemoryNode:
    """A single memory in the pyramid."""

    id: str
    content: str
    augmented_content: str
    embedding: list[float]
    effective_embedding: list[float]
    entities: dict[str, list[str]]
    R: float
    f: int
    t: int
    D: float
    M: float
    cube: CubeName
    created_at: int
    last_accessed: int
    M_history: list[float] = field(default_factory=list)
    cluster_id: int = -1

    def __post_init__(self) -> None:
        if self.cube not in _VALID_CUBES:
            raise ValueError(f"cube must be one of {_VALID_CUBES}, got {self.cube!r}")


@dataclass
class Theme:
    """A Level-3 cluster of related facts."""

    centroid: np.ndarray
    summary: str
    fact_ids: list[str]
    strength: float


@dataclass
class QueryResult:
    """Result of a pyramid query."""

    worldview: str = ""
    theme: str = ""
    facts: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    all_context: list[str] = field(default_factory=list)
