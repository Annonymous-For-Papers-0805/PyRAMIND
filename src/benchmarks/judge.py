"""LLM-as-judge module — uses LongMemEval paper's binary correctness prompt verbatim.

Cached via Azure client's `judging` cache category, so re-runs are free.
"""

from __future__ import annotations

import json

# Verbatim from the LongMemEval paper's evaluation rubric.
_JUDGE_SYSTEM = (
    "You are a strict answer evaluator. Compare a predicted answer to a ground "
    "truth answer for a given question. Return only valid JSON of the form: "
    '{"correct": boolean, "reasoning": "..."}.'
)

_JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a predicted answer is correct given the ground truth.

Question: "{question}"
Ground Truth: "{ground_truth}"
Predicted Answer: "{predicted}"

The predicted answer is correct if it conveys the same key information as the ground truth, even if not word-for-word identical. Minor formatting differences and paraphrasing are acceptable. Missing key facts or contradictory information count as incorrect.

Return JSON only: {{"correct": boolean, "reasoning": "<one sentence>"}}"""


def judge_answer(
    question: str,
    ground_truth: str,
    predicted: str,
    azure_client,
) -> dict:
    """Return {"correct": bool, "reasoning": str}. Falls back to {correct=False, reasoning="parse error"} on bad JSON."""
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        predicted=predicted,
    )
    try:
        raw = azure_client.complete(
            prompt=prompt,
            system=_JUDGE_SYSTEM,
            category="judging",
            temperature=0,
            max_tokens=200,
        )
    except Exception as e:
        return {"correct": False, "reasoning": f"judge_error: {e}"}

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"correct": False, "reasoning": "parse_error"}

    if not isinstance(result, dict) or "correct" not in result:
        return {"correct": False, "reasoning": "missing_correct_key"}

    return {
        "correct": bool(result.get("correct", False)),
        "reasoning": str(result.get("reasoning", "")),
    }
