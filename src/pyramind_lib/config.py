"""PyramindConfig — frozen-during-experiment configuration with 8 feature flags."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union


@dataclass
class PyramindConfig:
    """All hyperparameters and feature flags for the engine.

    Single source of truth — frozen for an entire experiment per the
    publishable-benchmark spec § 3.1. Hyperparameters chosen on a held-out
    validation slice and never tuned per test item.
    """

    # ── Required ──────────────────────────────────────────────────────────
    agent_role: str

    # ── Hyperparameters (frozen across an experiment, paper Table 1) ──────
    tau: float = 40.0
    tau_multiplier: float = 8.0
    promote_threshold: float = 0.30
    forget_threshold: float = 0.01
    pyramid_interval: int = 3
    embedding_drift_rate: float = 0.05

    # ── Paper-faithful R / density clamps (Eq. 2-3) ───────────────────────
    R_min: float = 0.05
    R_max: float = 0.95
    eps_D: float = 0.01
    warm_start_K: int = 8

    # ── Innovation B (deterministic R) ─────────────────────────────────────
    default_R: float = 0.5

    # ── Innovation D (momentum) ────────────────────────────────────────────
    momentum_boost: float = 1.2
    momentum_decay: float = 0.85
    momentum_history_len: int = 5

    # ── Innovation E (entropy adaptive forget) ─────────────────────────────
    entropy_high_threshold: float = 0.85
    entropy_low_threshold: float = 0.4
    entropy_high_multiplier: float = 1.5
    entropy_low_multiplier: float = 0.7

    # ── Innovation G (cross-reinforcement) ─────────────────────────────────
    cross_reinforce_threshold: float = 0.5
    cross_reinforce_max: int = 3

    # ── Engine internals ───────────────────────────────────────────────────
    dedup_threshold: float = 0.97
    min_promote_frequency: int = 2
    max_theme_clusters: int = 5
    embedding_model: str = "text-embedding-3-large"
    formula_name: str = "ensemble"

    # ── Embedding backend (paper default = "bge"; "azure" is cost-friendly) ─
    # The engine still embeds via AzureClient today; this field is the
    # source-of-truth flag that downstream code (and the embeddings module)
    # consult so paper-faithful runs can pick BGE without code edits.
    embedding_backend: str = "bge"

    # ── Tier-ablation flags (paper §4 ablation: which Pyramid tiers to emit) ─
    # Defaults are all on — the engine emits raw + facts + themes + worldview.
    # Turning any flag off suppresses the corresponding block in the query
    # output, which is what the per-tier ablation rows in the paper require.
    enable_facts: bool = True
    enable_themes: bool = True
    enable_worldview: bool = True

    # ── 8 innovation flags (A-H) — all ON for the headline result ──────────
    enable_context_vector: bool = True            # A
    enable_deterministic_R: bool = True           # B
    enable_pyramid: bool = True                   # C
    enable_momentum: bool = True                  # D
    enable_entropy_adaptive_forget: bool = True   # E
    enable_evolving_embeddings: bool = True       # F
    enable_cross_reinforcement: bool = True       # G
    enable_entity_augmentation: bool = True       # H

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if not 0.0 <= self.forget_threshold <= 1.0:
            raise ValueError(f"forget_threshold must be in [0, 1], got {self.forget_threshold}")
        if not 0.0 <= self.promote_threshold <= 1.0:
            raise ValueError(f"promote_threshold must be in [0, 1], got {self.promote_threshold}")
        if self.tau_multiplier <= 0:
            raise ValueError(f"tau_multiplier must be > 0, got {self.tau_multiplier}")

    def save(self, path: Union[Path, str]) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Union[Path, str]) -> "PyramindConfig":
        return cls(**json.loads(Path(path).read_text()))

    def frozen_dict(self) -> dict:
        """Hashable-ish dict representation for logging / cache keys."""
        return asdict(self)
