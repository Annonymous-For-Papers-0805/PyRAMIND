"""Engine innovations A-H as flag-gated pure functions.

Innovation C (pyramid compression) lives in `pyramid.py`.
Innovation B (deterministic R) lives in `scoring.py`.
Innovation H (entity augmentation) lives in `entities.py`.

This module covers A, D, E, F, G — the lifecycle-management innovations.
Each function is a no-op when its corresponding `enable_*` flag is OFF.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pyramind_lib.config import PyramindConfig
from pyramind_lib.types import MemoryNode


# ── Innovation A: Context vector ───────────────────────────────────────────

def update_context_vector(
    memories: list, config: PyramindConfig
) -> Optional[np.ndarray]:
    """M-weighted unit-normalized average of all effective embeddings.

    Returns None if disabled or no memories.
    """
    if not config.enable_context_vector or not memories:
        return None

    embeddings = np.array([m.effective_embedding for m in memories])
    weights = np.array([max(m.M, 0.001) for m in memories])
    weights = weights / weights.sum()
    ctx = np.average(embeddings, axis=0, weights=weights)
    norm = np.linalg.norm(ctx)
    if norm > 0:
        ctx = ctx / norm
    return ctx


# ── Innovation D: Momentum ─────────────────────────────────────────────────

def apply_momentum(memory: MemoryNode, config: PyramindConfig) -> float:
    """Boost or decay M based on recent M trend in M_history.

    Rising trend (last 3 strictly increasing) → multiply by momentum_boost.
    Falling trend (last 3 strictly decreasing) → multiply by momentum_decay.
    Otherwise unchanged.
    """
    if not config.enable_momentum:
        return memory.M
    history = memory.M_history
    if len(history) < 3:
        return memory.M
    if history[-1] > history[-2] > history[-3]:
        return memory.M * config.momentum_boost
    if history[-1] < history[-2] < history[-3]:
        return memory.M * config.momentum_decay
    return memory.M


# ── Innovation E: Entropy-adaptive forget threshold ────────────────────────

def compute_entropy(memories: list) -> float:
    """Shannon entropy of normalized M values across memories.

    Returns 0.0 if fewer than 2 memories.
    """
    if len(memories) < 2:
        return 0.0
    M_values = np.array([max(m.M, 0.001) for m in memories])
    probs = M_values / M_values.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def adaptive_forget_threshold(memories: list, config: PyramindConfig) -> float:
    """Scale the forget threshold by entropy.

    High entropy (M evenly distributed) → prune harder (× entropy_high_multiplier).
    Low entropy (M concentrated) → prune softer (× entropy_low_multiplier).
    Disabled → return base threshold unchanged.
    """
    base = config.forget_threshold
    if not config.enable_entropy_adaptive_forget or len(memories) < 2:
        return base
    entropy = compute_entropy(memories)
    max_entropy = np.log2(max(len(memories), 2))
    normalized = entropy / max_entropy if max_entropy > 0 else 0
    if normalized > config.entropy_high_threshold:
        return base * config.entropy_high_multiplier
    if normalized < config.entropy_low_threshold:
        return base * config.entropy_low_multiplier
    return base


# ── Innovation F: Evolving embeddings ──────────────────────────────────────

def evolve_embeddings(
    memories: list,
    context_vector: Optional[np.ndarray],
    config: PyramindConfig,
) -> None:
    """In-place: drift each memory's effective_embedding toward the context.

    Drift amount is proportional to current alignment with the context.
    Memories orthogonal or anti-aligned to the context drift less or not at all.
    No-op when disabled or context is None.
    """
    if not config.enable_evolving_embeddings or context_vector is None:
        return

    alpha = config.embedding_drift_rate
    for m in memories:
        eff = np.array(m.effective_embedding)
        alignment = float(np.dot(eff, context_vector))
        if alignment <= 0:
            continue
        drift = alpha * alignment
        new_eff = (1 - drift) * eff + drift * context_vector
        norm = np.linalg.norm(new_eff)
        if norm > 0:
            new_eff = new_eff / norm
        m.effective_embedding = new_eff.tolist()


# ── Innovation G: Cross-reinforcement ──────────────────────────────────────

def cross_reinforce(
    new_embedding,
    memories: list,
    config: PyramindConfig,
    cycle_count: int,
    formula_fn,
) -> None:
    """In-place: reinforce existing memories that resemble the new one.

    For up to `cross_reinforce_max` existing memories with similarity above
    `cross_reinforce_threshold`, increment f and partially refresh t. Recompute
    M with the active formula.
    No-op when disabled or no memories.
    """
    if not config.enable_cross_reinforcement or not memories:
        return

    new_emb = np.array(new_embedding)
    scores = []
    for m in memories:
        sim = float(np.dot(new_emb, np.array(m.effective_embedding)))
        if sim > config.cross_reinforce_threshold:
            scores.append((m, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    tau = max(config.tau, 0.01)

    from pyramind_lib.types import FormulaInputs

    for m, _sim in scores[: config.cross_reinforce_max]:
        m.f += 1
        m.t = max(0, m.t - 2)
        m.last_accessed = cycle_count
        eff_tau = tau * max(config.tau_multiplier, 0.01) if m.cube == "fact" else tau
        m.M = formula_fn(FormulaInputs(R=m.R, f=m.f, t=m.t, tau=eff_tau, D=m.D))
