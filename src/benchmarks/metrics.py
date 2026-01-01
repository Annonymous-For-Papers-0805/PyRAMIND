"""Metrics: F1, accuracy, recall@k, bootstrap CI, McNemar's test, Wilson CI."""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def compute_f1(predicted: str, ground_truth: str) -> float:
    """Token-overlap F1 (case-insensitive, whitespace-tokenized)."""
    pred_tokens = set(predicted.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    if not pred_tokens or not truth_tokens:
        return 0.0
    overlap = len(pred_tokens & truth_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_accuracy(results: Sequence) -> float:
    """Fraction of results with correct=True. Each result is a dict with key 'correct'."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.get("correct"))
    return correct / len(results)


def compute_recall_at_k(retrieved: Sequence, relevant: Sequence, k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved."""
    if not relevant:
        return 0.0
    top_k = list(retrieved)[:k]
    relevant_set = set(relevant)
    found = sum(1 for r in top_k if r in relevant_set)
    return found / len(relevant_set)


def bootstrap_ci(
    values: Sequence, B: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple:
    """Paired bootstrap CI for mean of values. Returns (mean, lower, upper)."""
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = []
    for _ in range(B):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(float(np.mean(sample)))
    means.sort()
    lower = means[int(B * alpha / 2)]
    upper = means[int(B * (1 - alpha / 2))]
    return (float(np.mean(arr)), lower, upper)


def mcnemar_test(
    system_a_correct: Sequence, system_b_correct: Sequence
) -> dict:
    """McNemar's exact test on paired binary outcomes.

    Inputs: two equal-length sequences of 0/1 (or False/True) — one per system,
    aligned by question index.

    Returns: {"b_only": int, "a_only": int, "p_value": float, "significant_at_05": bool}
    where:
      - a_only = #questions where A correct, B incorrect
      - b_only = #questions where B correct, A incorrect

    Uses exact binomial test (mid-p adjustment) on the (a_only, b_only) pair.
    """
    if len(system_a_correct) != len(system_b_correct):
        raise ValueError("Inputs must be the same length")

    a_only = 0
    b_only = 0
    for a, b in zip(system_a_correct, system_b_correct):
        if a and not b:
            a_only += 1
        elif b and not a:
            b_only += 1

    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0, "significant_at_05": False}

    # Exact two-sided binomial test: P(X <= min(a_only, b_only)) under H0 (p=0.5)
    k = min(a_only, b_only)
    p_one = 0.0
    for i in range(k + 1):
        p_one += math.comb(n, i) * (0.5 ** n)
    p_value = min(1.0, 2 * p_one)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
    }


# Back-compat alias used by the older tests / scripts that imported the
# exact-test under this name.
mcnemar_exact_test = mcnemar_test


def mcnemar_chi2(a: List[int], b: List[int]) -> dict:
    """McNemar's chi-squared with Yates continuity correction on paired correctness.

    ``a``, ``b`` are equal-length 0/1 vectors of per-item correctness for two
    systems on the same questions. Returns a dict with:

      - ``chi2``   — the corrected chi-squared statistic with 1 df
      - ``p``      — one-sided p-value from the chi-squared survival fn
                    (``erfc(sqrt(chi2/2))``); see paper §4.
      - ``a_only`` — count where A correct & B wrong (discordant pairs A>B)
      - ``b_only`` — count where B correct & A wrong (discordant pairs B>A)

    When the discordant total is 0 the test is undefined; we return chi2=0,
    p=1.0 (no evidence of a difference).
    """
    from math import erfc, sqrt
    assert len(a) == len(b)
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = a_only + b_only
    if n == 0:
        return {"chi2": 0.0, "p": 1.0, "a_only": a_only, "b_only": b_only}
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n
    # chi2 with 1 df: p = erfc(sqrt(chi2/2))
    p = erfc(sqrt(chi2 / 2.0))
    return {"chi2": chi2, "p": p, "a_only": a_only, "b_only": b_only}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion.

    More accurate than the normal-approximation interval at small ``n`` and
    when the observed proportion is near 0 or 1. Defaults to 95% (z=1.96).
    Returns ``(lower, upper)``; degenerate ``n==0`` yields ``(0.0, 1.0)``.
    """
    from math import sqrt
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = (z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))
