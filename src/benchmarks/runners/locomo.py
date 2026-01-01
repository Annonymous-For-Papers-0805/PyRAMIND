"""LoCoMo runner — same shape as LME runner, iterates conversations × QA pairs."""

from __future__ import annotations

import time

from benchmarks.judge import judge_answer
from benchmarks.metrics import compute_f1
from benchmarks.runners.lme import QuestionRecord, RunResult


def run_locomo(
    baseline,
    items: list,
    judge_client,
    seed: int = 42,
) -> RunResult:
    """Run a baseline over LoCoMo conversations.

    For each conversation: reset baseline → ingest all sessions in order →
    answer each QA pair with the populated baseline.
    """
    result = RunResult(system=baseline.name, benchmark="locomo", seed=seed)
    t0 = time.time()

    for ci, item in enumerate(items):
        baseline.reset()
        messages = item.flatten_messages()
        if messages:
            baseline.ingest(messages)

        for qi, qa in enumerate(item.qa):
            q_start = time.time()
            try:
                predicted = baseline.query(qa.question)
            except Exception as e:
                predicted = f"[ERROR: {e}]"
            elapsed_ms = int((time.time() - q_start) * 1000)

            verdict = judge_answer(qa.question, qa.answer, predicted, judge_client)
            f1 = compute_f1(predicted, qa.answer)
            retrieved = list(getattr(baseline, "last_retrieved", []) or [])

            result.records.append(
                QuestionRecord(
                    question_id=f"locomo-{item.conversation_id}-{qi}",
                    question_type=f"category-{qa.category}",
                    question=qa.question,
                    ground_truth=qa.answer,
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
