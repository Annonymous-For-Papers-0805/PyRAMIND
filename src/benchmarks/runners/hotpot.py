"""HotpotQA runner — drives a Baseline through one Hotpot pass.

Each item is independent (per-question reset). The baseline ingests the 10
context paragraphs, then answers the question. F1 is computed against the
ground-truth answer; the LLM judge applies the same yes/no protocol used
elsewhere.
"""

from __future__ import annotations

import time

from benchmarks.judge import judge_answer
from benchmarks.metrics import compute_f1
from benchmarks.runners.lme import QuestionRecord, RunResult


def run_hotpot(
    baseline,
    items: list,
    judge_client,
    seed: int = 42,
) -> RunResult:
    result = RunResult(system=baseline.name, benchmark="hotpot", seed=seed)
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
                question_type=f"{item.qtype}-{item.level}",
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
