"""Aggregator — reads run artifacts and produces per-cell summary stats.

Walks a directory tree of orchestrator artifacts in the canonical layout

    runs/
        <benchmark>/
            <system>/
                seed-<N>.json

and produces:

  - per-(benchmark, system) cell stats: mean accuracy ± std across seeds, n_seeds
  - per-cell concatenated per-question records (for downstream significance tests)
  - cost rollups (sum across seeds + means)

The cell-level accuracy is the *mean of per-seed accuracies*, not a pooled
accuracy across all questions, because seeds are paired across systems and
treating questions as i.i.d. across seeds would inflate n.

Reference: design spec §6 (results aggregation).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


# ── data classes ────────────────────────────────────────────────────────────


@dataclass
class RunArtifact:
    """One per-(system, benchmark, seed) run artifact loaded from JSON."""
    path: Path
    system: str
    benchmark: str
    seed: int
    config_hash: str
    accuracy: float
    avg_f1: float
    n_questions: int
    avg_latency_ms: int
    wall_time_seconds: float
    cost: dict
    records: List[dict]

    @classmethod
    def from_json(cls, path: Path) -> "RunArtifact":
        payload = json.loads(Path(path).read_text())
        s = payload["summary"]
        return cls(
            path=Path(path),
            system=payload["system"],
            benchmark=payload["benchmark"],
            seed=int(payload["seed"]),
            config_hash=payload.get("config_hash", ""),
            accuracy=float(s["accuracy"]),
            avg_f1=float(s["avg_f1"]),
            n_questions=int(s["n"]),
            avg_latency_ms=int(s.get("avg_latency_ms", 0)),
            wall_time_seconds=float(payload.get("wall_time_seconds", 0.0)),
            cost=dict(payload.get("cost", {})),
            records=list(payload.get("records", [])),
        )


@dataclass
class CellStats:
    """Aggregated statistics for one (benchmark, system) cell across seeds."""
    benchmark: str
    system: str
    n_seeds: int
    seeds: List[int]
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    std_f1: float
    mean_latency_ms: float
    total_questions: int
    total_cost_tokens: int
    # Wilson 95% CI on the pooled per-question correctness across all
    # seeds; reported per cell in the summary printout (paper §4).
    wilson_ci_95: Tuple[float, float] = (0.0, 1.0)
    artifacts: List[RunArtifact] = field(default_factory=list)

    def to_summary_dict(self) -> dict:
        d = asdict(self)
        d.pop("artifacts", None)
        return d


@dataclass
class AggregateResult:
    cells: Dict[Tuple[str, str], CellStats]
    """(benchmark, system) → CellStats."""

    def benchmarks(self) -> List[str]:
        return sorted({b for b, _ in self.cells.keys()})

    def systems(self) -> List[str]:
        return sorted({s for _, s in self.cells.keys()})

    def cell(self, benchmark: str, system: str) -> Optional[CellStats]:
        return self.cells.get((benchmark, system))

    def to_summary_dict(self) -> dict:
        return {
            "benchmarks": self.benchmarks(),
            "systems": self.systems(),
            "cells": {
                f"{b}/{s}": stats.to_summary_dict()
                for (b, s), stats in self.cells.items()
            },
        }


# ── loading ─────────────────────────────────────────────────────────────────


def discover_artifacts(root: Union[str, Path]) -> List[Path]:
    """Find all seed-*.json artifacts under root, recursively."""
    return sorted(Path(root).rglob("seed-*.json"))


def load_artifacts(root: Union[str, Path]) -> List[RunArtifact]:
    """Load every artifact under root, skipping malformed ones with a warning."""
    out: List[RunArtifact] = []
    for p in discover_artifacts(root):
        try:
            out.append(RunArtifact.from_json(p))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # Don't crash the entire aggregation on one bad file.
            print(f"[aggregate] WARN skipping malformed artifact {p}: {e}")
    return out


# ── aggregation ─────────────────────────────────────────────────────────────


def _safe_stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs)


def aggregate(artifacts: Iterable[RunArtifact]) -> AggregateResult:
    """Group artifacts by (benchmark, system) and compute per-cell stats."""
    grouped: Dict[Tuple[str, str], List[RunArtifact]] = {}
    for a in artifacts:
        grouped.setdefault((a.benchmark, a.system), []).append(a)

    from benchmarks.metrics import wilson_ci

    cells: Dict[Tuple[str, str], CellStats] = {}
    for (benchmark, system), arts in grouped.items():
        arts_sorted = sorted(arts, key=lambda x: x.seed)
        accs = [a.accuracy for a in arts_sorted]
        f1s = [a.avg_f1 for a in arts_sorted]
        lats = [a.avg_latency_ms for a in arts_sorted]
        total_q = sum(a.n_questions for a in arts_sorted)
        total_cost = sum(int(a.cost.get("total", 0)) for a in arts_sorted)

        # Wilson 95% CI on the pooled per-question correctness vector
        # across seeds. Treats each (seed, question) outcome as a single
        # Bernoulli draw — looser than a paired test but a reasonable
        # summary-table CI (paper §4).
        n_correct = 0
        n_total = 0
        for art in arts_sorted:
            for r in art.records:
                n_total += 1
                if bool(r.get("correct", False)):
                    n_correct += 1
        ci_lo, ci_hi = wilson_ci(n_correct, n_total)

        cells[(benchmark, system)] = CellStats(
            benchmark=benchmark, system=system,
            n_seeds=len(arts_sorted),
            seeds=[a.seed for a in arts_sorted],
            mean_accuracy=statistics.mean(accs),
            std_accuracy=_safe_stdev(accs),
            mean_f1=statistics.mean(f1s),
            std_f1=_safe_stdev(f1s),
            mean_latency_ms=statistics.mean(lats) if lats else 0.0,
            total_questions=total_q,
            total_cost_tokens=total_cost,
            wilson_ci_95=(round(ci_lo, 4), round(ci_hi, 4)),
            artifacts=arts_sorted,
        )
    return AggregateResult(cells=cells)


def aggregate_dir(root: Union[str, Path]) -> AggregateResult:
    """Convenience: discover + load + aggregate in one call."""
    return aggregate(load_artifacts(root))


# ── slicing helpers ─────────────────────────────────────────────────────────


def slice_records_by_question_type(
    artifact: RunArtifact, prefix: Optional[str] = None,
) -> List[dict]:
    """Return per-question records optionally filtered by question_type prefix."""
    if prefix is None:
        return list(artifact.records)
    return [r for r in artifact.records if str(r.get("question_type", "")).startswith(prefix)]


def cell_per_question_correctness(
    cell: CellStats, *, question_type_prefix: Optional[str] = None,
) -> List[bool]:
    """Flatten correctness flags across all seeds for one cell.

    Used by significance tests. Records preserve their per-seed order; the
    same question_id should appear once per seed at the same offset.
    """
    out: List[bool] = []
    for art in cell.artifacts:
        for r in slice_records_by_question_type(art, question_type_prefix):
            out.append(bool(r.get("correct", False)))
    return out


def cell_per_question_correctness_paired(
    cell_a: CellStats, cell_b: CellStats,
) -> Tuple[List[bool], List[bool]]:
    """Return (a_correct, b_correct) aligned by (seed, question_id).

    Two systems on the same benchmark share question IDs; we pair them up by
    (seed, question_id) so McNemar's test sees genuinely paired observations.
    Returns only IDs that appear in BOTH cells for any given seed.
    """
    a_by_seed: Dict[int, Dict[str, bool]] = {}
    for art in cell_a.artifacts:
        a_by_seed[art.seed] = {
            str(r["question_id"]): bool(r["correct"]) for r in art.records
        }
    b_by_seed: Dict[int, Dict[str, bool]] = {}
    for art in cell_b.artifacts:
        b_by_seed[art.seed] = {
            str(r["question_id"]): bool(r["correct"]) for r in art.records
        }

    a_out: List[bool] = []
    b_out: List[bool] = []
    for seed in sorted(set(a_by_seed) & set(b_by_seed)):
        a_map, b_map = a_by_seed[seed], b_by_seed[seed]
        for qid in sorted(set(a_map) & set(b_map)):
            a_out.append(a_map[qid])
            b_out.append(b_map[qid])
    return a_out, b_out


# ── CLI ────────────────────────────────────────────────────────────────────


def write_summary(result: AggregateResult, out_path: Union[str, Path]) -> Path:
    """Dump a JSON summary suitable for downstream report rendering."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_summary_dict(), indent=2))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="benchmarks.aggregate",
        description="Aggregate run artifacts into per-cell summary stats.",
    )
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Where to write the summary JSON.")
    args = p.parse_args(argv)

    result = aggregate_dir(args.runs_dir)
    out = write_summary(result, args.out)
    print(
        f"[aggregate] {len(result.cells)} cells across "
        f"{len(result.benchmarks())} benchmarks × {len(result.systems())} systems"
    )
    # Per-cell printout with Wilson 95% CI on pooled per-question correctness.
    for (b, s), stats in sorted(result.cells.items()):
        lo, hi = stats.wilson_ci_95
        print(
            f"  {b}/{s}: acc={stats.mean_accuracy:.3f} "
            f"(95% Wilson CI [{lo:.3f}, {hi:.3f}], "
            f"n={stats.total_questions}, seeds={stats.n_seeds})"
        )
    print(f"[aggregate] wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
