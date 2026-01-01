"""Memory-decay formulas: classical baselines + the PyRAMIND ensemble.

Each formula maps a FormulaInputs(R, f, t, tau, D) tuple to a scalar
memory-strength score. The PyRAMIND memory engine uses the upper envelope
of three forgetting-curve estimators (Eq. 4 in the paper):

    M_ensemble = max(M_pyramind, M_power_law, M_bayesian)

where M_pyramind is the canonical Ebbinghaus-style exponential with
log-frequency reinforcement, M_power_law is the Wickelgren-Wixted power
law, and M_bayesian is the Anderson rational-analysis posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from pyramind_lib.types import FormulaInputs


# ---- canonical PyRAMIND formula (Eq. 1) -----------------------------------

def _pyramind_decay(fi: FormulaInputs) -> float:
    """PyRAMIND canonical decay: M = R * ln(f+1) * exp(-t / (tau * D)).

    Ebbinghaus-style exponential modulated by log-frequency reinforcement
    and the local-density factor D in (0, 1].
    """
    if fi.R == 0:
        return 0.0
    f = max(fi.f, 1)
    return fi.R * math.log(f + 1) * math.exp(-fi.t / (fi.tau * fi.D))


# ---- classical baselines -------------------------------------------------

def _ebbinghaus(fi: FormulaInputs) -> float:
    """Ebbinghaus 1885: R = exp(-t / S), with S = tau * (1 + 0.5 * ln(f+1))."""
    f = max(fi.f, 1)
    S = fi.tau * (1.0 + 0.5 * math.log(f + 1))
    return fi.R * math.exp(-fi.t / max(S, 1e-9))


def _power_law(fi: FormulaInputs) -> float:
    """Wickelgren-Wixted power law (paper Eq. 5):
    M_pow = R * ln(f+1) * (t+1)^(-1 / (tau * D)).
    """
    if fi.R == 0:
        return 0.0
    f = max(fi.f, 1)
    b = 1.0 / (fi.tau * fi.D)
    return fi.R * math.log(f + 1) * math.pow(fi.t + 1.0, -b)


def _actr(fi: FormulaInputs) -> float:
    """ACT-R: B = ln(n * t^(-d))."""
    f = max(fi.f, 1)
    if fi.t <= 0:
        return fi.R * math.log(f + 1)
    d = 0.5
    activation = math.log(f * math.pow(max(fi.t, 0.1), -d))
    normalized = max(0.0, (activation + 3.0) / 4.0)
    return fi.R * normalized


def _fsrs(fi: FormulaInputs) -> float:
    """FSRS 2023: R = (1 + t / 9S)^(-0.5)."""
    f = max(fi.f, 1)
    factor = 19.0 / 81.0
    decay = 0.5
    S = fi.tau * math.pow(f, 0.4)
    if fi.t <= 0:
        return fi.R * math.log(f + 1)
    retention = math.pow(1 + factor * fi.t / max(S, 1e-9), -decay)
    return fi.R * math.log(f + 1) * retention


def _stanford(fi: FormulaInputs) -> float:
    """Park et al. 2023: weighted recency + relevance + density."""
    alpha, beta, gamma = 1.0, 1.0, 0.5
    recency = math.exp(-fi.t / fi.tau)
    return (alpha * recency + beta * fi.R + gamma * fi.D) / (alpha + beta + gamma)


def _fade_mem(fi: FormulaInputs) -> float:
    """FadeMem: I = alpha * rel + beta * freq + gamma * rec."""
    f = max(fi.f, 1)
    alpha, beta, gamma = 0.4, 0.3, 0.3
    relevance = fi.R * fi.D
    freq_term = math.log(f + 1) / math.log(10)
    recency = math.exp(-fi.t / fi.tau)
    return alpha * relevance + beta * freq_term + gamma * recency


def _bayesian(fi: FormulaInputs) -> float:
    """Anderson 1990: posterior ~ prior * likelihood * decay."""
    f = max(fi.f, 1)
    prior = 0.5
    likelihood = fi.R * (1.0 - math.exp(-f * 0.5))
    decay_prior = math.exp(-fi.t / (fi.tau * fi.D))
    num = prior * likelihood * decay_prior
    den = num + ((1 - prior) * (1 - likelihood) * (1 - decay_prior) + 0.001)
    posterior = num / den
    return max(0.0, min(1.5, posterior * math.log(f + 1)))


def _hebbian(fi: FormulaInputs) -> float:
    """Hebb 1949: w = w0 * (1 + eta * f) * exp(-lambda * t)."""
    f = max(fi.f, 1)
    eta = 0.3
    return fi.R * (1.0 + eta * f) * math.exp(-fi.t / fi.tau)


# ---- innovations ---------------------------------------------------------

def _bayesian_act_r(fi: FormulaInputs) -> float:
    """ACT-R Bayesian posterior: pi(R, f) * ln(f+1) * exp(-t / (tau * D))."""
    if fi.R == 0:
        return 0.0
    f = max(fi.f, 1)
    prior = 0.5
    likelihood = fi.R * (1.0 - math.exp(-f * 0.5))
    decay_prior = math.exp(-fi.t / (fi.tau * fi.D))
    num = prior * likelihood * decay_prior
    den = num + ((1 - prior) * (1 - likelihood) * (1 - decay_prior) + 0.001)
    posterior = num / den
    return posterior * math.log(f + 1) * math.exp(-fi.t / (fi.tau * fi.D))


def _actr_density(fi: FormulaInputs) -> float:
    """Sum-of-traces ACT-R activation with density modulation."""
    f = max(fi.f, 1)
    if fi.t <= 0:
        return fi.R * math.log(f + 1) * fi.D
    d = 0.5
    trace_sum = 0.0
    for j in range(f):
        trace_age = max(0.1, fi.t - j * (fi.t / f))
        trace_sum += math.pow(trace_age, -d)
    activation = math.log(max(0.001, trace_sum))
    normalized = max(0.0, (activation + 3.0) / 4.0)
    return fi.R * normalized * fi.D


def _adaptive_tau(fi: FormulaInputs) -> float:
    """tau adapts per-memory: tau_eff = tau * (1 + 0.2 * f) * D."""
    if fi.R == 0:
        return 0.0
    f = max(fi.f, 1)
    tau_eff = fi.tau * (1.0 + 0.2 * f) * fi.D
    return fi.R * math.log(f + 1) * math.exp(-fi.t / tau_eff)


def _ensemble(fi: FormulaInputs) -> float:
    """PyRAMIND ensemble: max(pyramind_decay, power_law, bayesian_act_r)."""
    return max(_pyramind_decay(fi), _power_law(fi), _bayesian_act_r(fi))


def _ensemble_mean(fi: FormulaInputs) -> float:
    """Ensemble variant — arithmetic mean of the three estimators.

    Smoother than the upper-envelope ``max`` ensemble; trades a small bias
    for lower variance across noisy R / D estimates. Reported in the paper
    as an ablation row.
    """
    return (_pyramind_decay(fi) + _power_law(fi) + _bayesian_act_r(fi)) / 3.0


def _ensemble_median(fi: FormulaInputs) -> float:
    """Ensemble variant — median of the three estimators.

    Robust-statistic variant that ignores the single most-extreme component
    score, useful when one estimator pathologically saturates (e.g.
    Bayesian-ACT-R near f=1).
    """
    vals = sorted([_pyramind_decay(fi), _power_law(fi), _bayesian_act_r(fi)])
    return vals[1]


# ---- registry ------------------------------------------------------------

FORMULAS: Dict[str, Callable[[FormulaInputs], float]] = {
    "pyramind":        _pyramind_decay,      # canonical (Eq. 1)
    "ebbinghaus":      _ebbinghaus,
    "power-law":       _power_law,
    "actr":            _actr,
    "fsrs":            _fsrs,
    "stanford":        _stanford,
    "fade-mem":        _fade_mem,
    "bayesian":        _bayesian,
    "hebbian":         _hebbian,
    "bayesian-act-r":  _bayesian_act_r,
    "actr-density":    _actr_density,
    "adaptive-tau":    _adaptive_tau,
    "ensemble":        _ensemble,            # PyRAMIND default (Eq. 4)
    "ensemble-mean":   _ensemble_mean,       # ablation: arithmetic mean
    "ensemble-median": _ensemble_median,     # ablation: median (robust)
}


FORMULA_INFO = {
    "pyramind":       {"id": "pyramind",       "name": "PyRAMIND",         "expression": "M_base = R*ln(f+1)*e^(-t/τD)",      "type": "component"},
    "ebbinghaus":     {"id": "ebbinghaus",     "name": "Ebbinghaus",      "expression": "R = e^(-t/S)",                     "type": "classical"},
    "power-law":      {"id": "power-law",      "name": "Power-Law",       "expression": "M = R*ln(f+1)*t^(-1/τD)",          "type": "classical"},
    "actr":           {"id": "actr",           "name": "ACT-R",           "expression": "B = ln(n*t^(-d))",                 "type": "classical"},
    "fsrs":           {"id": "fsrs",           "name": "FSRS",            "expression": "R = (1+t/9S)^(-0.5)",              "type": "classical"},
    "stanford":       {"id": "stanford",       "name": "Stanford",        "expression": "α*rec + β*imp + γ*rel",            "type": "classical"},
    "fade-mem":       {"id": "fade-mem",       "name": "FadeMem",         "expression": "α*rel + β*freq + γ*rec",           "type": "classical"},
    "bayesian":       {"id": "bayesian",       "name": "Bayesian",        "expression": "P ∝ prior*like*decay",             "type": "classical"},
    "hebbian":        {"id": "hebbian",        "name": "Hebbian",         "expression": "w = w0*(1+ηf)*e^(-λt)",            "type": "classical"},
    "bayesian-act-r": {"id": "bayesian-act-r", "name": "Bayesian-ACT-R",  "expression": "π(R,f)*ln(f+1)*e^(-t/τD)",         "type": "innovation"},
    "actr-density":   {"id": "actr-density",   "name": "ACT-R+Density",   "expression": "ln(Σt_j^(-d)) * D",                "type": "innovation"},
    "adaptive-tau":   {"id": "adaptive-tau",   "name": "Adaptive-τ",      "expression": "τ_eff = τ*(1+0.2f)*D",             "type": "innovation"},
    "ensemble":       {"id": "ensemble",       "name": "Ensemble",        "expression": "max(pyramind, power-law, bayesian)","type": "ensemble"},
    "ensemble-mean":  {"id": "ensemble-mean",  "name": "Ensemble-Mean",   "expression": "mean(pyramind, power-law, bayesian)","type": "ensemble"},
    "ensemble-median":{"id": "ensemble-median","name": "Ensemble-Median", "expression": "median(pyramind, power-law, bayesian)","type": "ensemble"},
}


def get_formula(name: str):
    if name not in FORMULAS:
        raise ValueError(
            f"Unknown formula: {name!r}. Available: {sorted(FORMULAS.keys())}"
        )
    return FORMULAS[name]


@dataclass
class FormulaRegistry:
    """Iterable view of all formulas (for benchmark runners)."""

    def all(self) -> list:
        return list(FORMULA_INFO.values())

    def by_type(self, type_: str) -> list:
        return [info for info in FORMULA_INFO.values() if info["type"] == type_]
