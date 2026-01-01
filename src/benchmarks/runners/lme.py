"""LongMemEval-S runner — drives a Baseline through one LME pass."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from benchmarks.judge import judge_answer
from benchmarks.metrics import compute_f1


@dataclass
class QuestionRecord:
    question_id: str
    question_type: str
    question: str
    ground_truth: str
    predicted: str
    correct: bool
    f1: float
    elapsed_ms: int
    judge_reasoning: str = ""
    retrieved: list = field(default_factory=list)  # for recall@k computation


@dataclass
class RunResult:
    system: str
    benchmark: str
    seed: int
    records: list = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    elapsed_total_seconds: float = 0.0

    def accuracy(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.correct) / len(self.records)

    def avg_f1(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.f1 for r in self.records) / len(self.records)


def run_lme(
    baseline,
    items: list,
    judge_client,
    seed: int = 42,
    benchmark_name: str = "lme",
) -> RunResult:
    """Run a baseline over LME items, returning per-question records.

    For each item: reset baseline → ingest the item's haystack messages →
    query with the item's question → judge against ground truth.

    `benchmark_name` lets the orchestrator route LME-S vs LME-Oracle to the
    same runner while preserving the dataset label in the result artifact.
    """
    result = RunResult(system=baseline.name, benchmark=benchmark_name, seed=seed)
    t0 = time.time()

    for item in items:
        baseline.reset()
        messages = item.flatten_messages()
        if messages:
            baseline.ingest(messages)

        q_start = time.time()
        try:
            predicted = baseline.query(item.question)
        except Exception as e:
            predicted = f"[ERROR: {e}]"
        elapsed_ms = int((time.time() - q_start) * 1000)

        verdict = judge_answer(item.question, item.answer, predicted, judge_client)
        f1 = compute_f1(predicted, item.answer)

        retrieved = list(getattr(baseline, "last_retrieved", []) or [])

        result.records.append(
            QuestionRecord(
                question_id=item.question_id,
                question_type=item.question_type,
                question=item.question,
                ground_truth=item.answer,
                predicted=predicted,
                correct=bool(verdict["correct"]),
                f1=f1,
                elapsed_ms=elapsed_ms,
                judge_reasoning=verdict.get("reasoning", ""),
                retrieved=retrieved,
            )
        )

    result.elapsed_total_seconds = time.time() - t0
    result.cost = baseline.cost_so_far()
    return result
