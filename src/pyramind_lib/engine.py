"""MemoryEngine — the orchestrator that wires all 8 innovations together.

Each lifecycle step (record, run_cycle, query) consults the relevant feature
flags in `PyramindConfig`. With every flag OFF, the engine reduces to a naive
embedding store — useful as the ablation floor.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Optional

import numpy as np

from pyramind_lib.config import PyramindConfig
from pyramind_lib.entities import augment_text, extract_entities
from pyramind_lib.formulas import FORMULAS
from pyramind_lib.innovations import (
    adaptive_forget_threshold,
    apply_momentum,
    cross_reinforce,
    evolve_embeddings,
    update_context_vector,
)
from pyramind_lib.pyramid import rebuild
from pyramind_lib.retrieval import cosine_similarity
from pyramind_lib.scoring import compute_R
from pyramind_lib.types import FormulaInputs, MemoryNode, QueryResult, Theme


class MemoryEngine:
    """Formula-agnostic memory engine with all 8 innovations as feature flags."""

    def __init__(
        self,
        config: PyramindConfig,
        azure_client,
        formula_name: Optional[str] = None,
    ) -> None:
        self.config = config
        self.azure_client = azure_client
        formula_name = formula_name or config.formula_name
        if formula_name not in FORMULAS:
            raise ValueError(
                f"Unknown formula: {formula_name!r}. Available: {sorted(FORMULAS.keys())}"
            )
        self.formula_fn = FORMULAS[formula_name]

        self.memories: list = []
        self.themes: list = []
        self.worldview_vector: Optional[np.ndarray] = None
        self.worldview_summary: str = ""
        self.context_vector: Optional[np.ndarray] = None
        self.cycle_count: int = 0

    # ── Helpers ────────────────────────────────────────────────────────────

    def _calculate_density(self, embedding) -> float:
        # Paper §3 warm-start: the first K writes use D_m ≡ 1 to avoid
        # empty / sparse neighbourhoods skewing the local-density estimate.
        if len(self.memories) < self.config.warm_start_K:
            return 1.0
        max_sim = max(
            cosine_similarity(embedding, m.effective_embedding) for m in self.memories
        )
        # Paper Eq. 3 clamps D to >= eps_D here, NOT inside every formula.
        return max(self.config.eps_D, 1.0 - max_sim)

    def _compute_M(self, R: float, f: int, t: int, tau: float, D: float) -> float:
        return self.formula_fn(FormulaInputs(R=R, f=f, t=t, tau=tau, D=D))

    # ── Record ─────────────────────────────────────────────────────────────

    def record(self, content: str) -> Optional[MemoryNode]:
        """Ingest one message. Returns the new MemoryNode (or the reinforced existing one)."""
        # Innovation H — entity augmentation (gated)
        if self.config.enable_entity_augmentation:
            entities = extract_entities(content, self.azure_client)
            augmented = augment_text(content, entities)
        else:
            entities = {"dates": [], "numbers": [], "names": [], "places": [], "topics": []}
            augmented = content

        embedding = self.azure_client.embed(augmented)
        tau = max(self.config.tau, 0.01)

        # Dedup against effective embeddings
        for existing in self.memories:
            sim = cosine_similarity(embedding, existing.effective_embedding)
            if sim > self.config.dedup_threshold:
                existing.f += 1
                existing.t = 0
                existing.last_accessed = self.cycle_count
                existing.M = self._compute_M(existing.R, existing.f, 0, tau, existing.D)
                return existing

        # Innovation G — cross-reinforcement (gated)
        cross_reinforce(
            embedding, self.memories, self.config, self.cycle_count, self.formula_fn
        )

        # Innovation B — deterministic R (gated inside compute_R)
        R = compute_R(content, embedding, self.context_vector, self.config)
        D = self._calculate_density(embedding)
        M = self._compute_M(R, 1, 0, tau, D)

        node = MemoryNode(
            id=str(uuid.uuid4()),
            content=content,
            augmented_content=augmented,
            embedding=embedding,
            effective_embedding=list(embedding),
            entities=entities,
            R=R, f=1, t=0, D=D, M=M,
            cube="raw",
            created_at=self.cycle_count,
            last_accessed=self.cycle_count,
            M_history=[M],
        )
        self.memories.append(node)

        # Paper §3: promotion raw → fact happens later in run_cycle when
        # M_m > θ_p AND f_m ≥ f_min. Do NOT pre-duplicate a fact node here.

        # Innovation A — context vector update (gated)
        self.context_vector = update_context_vector(self.memories, self.config)

        return node

    # ── Decay cycle ────────────────────────────────────────────────────────

    def run_cycle(self) -> None:
        """Age all memories one step, decay, evolve, prune, promote."""
        self.cycle_count += 1
        tau = max(self.config.tau, 0.01)
        tau_multiplier = max(self.config.tau_multiplier, 0.01)

        # Decay (always on). Use the same effective tau for raw and fact —
        # the literal raw text is the ground truth and should not decay
        # faster than the paraphrased fact derived from it.
        for m in self.memories:
            m.t += 1
            eff_tau = tau * tau_multiplier
            m.M = self._compute_M(m.R, m.f, m.t, eff_tau, m.D)
            m.M = apply_momentum(m, self.config)
            m.M_history.append(m.M)
            if len(m.M_history) > self.config.momentum_history_len:
                m.M_history = m.M_history[-self.config.momentum_history_len :]

        # Innovation F — evolve embeddings toward context (gated)
        evolve_embeddings(self.memories, self.context_vector, self.config)

        # Innovation E — adaptive forget threshold (gated inside).
        # Critical: NEVER prune raw memories. Raw is the ground-truth
        # episodic store; only paraphrased facts are eligible for forgetting.
        # This matches the design pattern in MemMachine / Mem0's ADD-only
        # algorithm where raw episodes are preserved indefinitely and only
        # the active retrieval pool is filtered.
        forget_threshold = adaptive_forget_threshold(self.memories, self.config)
        self.memories = [
            m for m in self.memories
            if m.cube == "raw" or m.M >= forget_threshold
        ]

        # Promote raw → fact (always on)
        for m in self.memories:
            if (
                m.cube == "raw"
                and m.M > self.config.promote_threshold
                and m.f >= self.config.min_promote_frequency
            ):
                m.cube = "fact"

        # Innovation C — pyramid rebuild (gated inside)
        if self.cycle_count % self.config.pyramid_interval == 0:
            facts = [m for m in self.memories if m.cube == "fact"]
            self.themes, self.worldview_vector, self.worldview_summary = rebuild(
                facts, self.config
            )

        # Innovation A — context vector refresh (gated)
        self.context_vector = update_context_vector(self.memories, self.config)

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 5) -> QueryResult:
        """Pyramid query — uses effective (evolved) embeddings."""
        q_emb = self.azure_client.embed(text)
        result = QueryResult()

        # Level 4 — Worldview (tier-ablation gate: enable_worldview).
        if (
            self.config.enable_worldview
            and self.worldview_vector is not None
            and self.worldview_summary
        ):
            result.worldview = self.worldview_summary
            result.all_context.append(f"[WORLDVIEW] {self.worldview_summary}")

        # Level 3 — Best theme (tier-ablation gate: enable_themes; also
        # requires the pyramid stage to be enabled and at least one theme).
        if (
            self.config.enable_themes
            and self.config.enable_pyramid
            and self.themes
        ):
            theme_scores = [
                (t, cosine_similarity(q_emb, t.centroid.tolist())) for t in self.themes
            ]
            theme_scores.sort(key=lambda x: x[1], reverse=True)
            best_theme = theme_scores[0][0]
            result.theme = best_theme.summary
            result.all_context.append(f"[THEME] {best_theme.summary}")

        # Level 2 — Facts (M-weighted cosine; tier-ablation gate: enable_facts).
        # FIX #3: use the FROZEN `embedding` (original text embedding) instead
        # of `effective_embedding` (which drifts toward the running context
        # vector at 5%/cycle and breaks retrieval against fresh question
        # embeddings).
        facts = (
            [m for m in self.memories if m.cube == "fact"]
            if self.config.enable_facts
            else []
        )
        if facts:
            fact_scores = [
                (f, cosine_similarity(q_emb, f.embedding) * max(f.M, 0.1) ** 0.3)
                for f in facts
            ]
            fact_scores.sort(key=lambda x: x[1], reverse=True)
            for f, _s in fact_scores[:top_k]:
                result.facts.append(f.content)
                result.all_context.append(f"[FACT] {f.content}")

        # Level 1 — Raw (cosine only). Surface a full top_k of raw memories
        # alongside the facts (not just leftover slots), so the answerer always
        # sees literal evidence even when the fact-tier rephrasing dropped a
        # critical token.
        raw = [m for m in self.memories if m.cube == "raw"]
        if raw:
            raw_scores = [
                (r, cosine_similarity(q_emb, r.embedding)) for r in raw
            ]
            raw_scores.sort(key=lambda x: x[1], reverse=True)
            for r, _s in raw_scores[:top_k]:
                result.raw.append(r.content)
                result.all_context.append(f"[MEMORY] {r.content}")

        # Mark accessed
        retrieved = set(result.facts) | set(result.raw)
        for m in self.memories:
            if m.content in retrieved:
                m.t = 0
                m.last_accessed = self.cycle_count

        return result

    # ── Stats / Reset / Determinism ────────────────────────────────────────

    def get_stats(self) -> dict:
        from pyramind_lib.innovations import compute_entropy

        raw_count = sum(1 for m in self.memories if m.cube == "raw")
        fact_count = sum(1 for m in self.memories if m.cube == "fact")
        return {
            "total": len(self.memories),
            "raw": raw_count,
            "facts": fact_count,
            "themes": len(self.themes),
            "has_worldview": self.worldview_vector is not None,
            "entropy": compute_entropy(self.memories),
        }

    def reset(self) -> None:
        self.memories = []
        self.themes = []
        self.worldview_vector = None
        self.worldview_summary = ""
        self.context_vector = None
        self.cycle_count = 0

    def memory_state_hash(self) -> str:
        """SHA-256 hash of canonicalized memory state — used for determinism tests."""
        canonical = []
        for m in sorted(self.memories, key=lambda x: x.created_at):
            canonical.append({
                "content": m.content,
                "f": m.f,
                "t": m.t,
                "cube": m.cube,
                "R": round(m.R, 6),
                "M": round(m.M, 6),
                "D": round(m.D, 6),
            })
        blob = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
