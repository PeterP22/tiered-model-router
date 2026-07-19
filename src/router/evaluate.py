"""Offline router evaluation on held-out Arena battles.

Quality proxy: for each cross-class test battle, the router (a score s(q) and
threshold t) picks a side — strong if s(q) >= t, else weak. Quality = fraction
of battles where the routed side is the side the human preferred. Cost =
fraction of queries routed to the strong model. Sweeping t traces the
cost/quality curve; PGR/APGR/CPT follow RouteLLM's definitions
(docs/phase1-research.md #5).
"""

from __future__ import annotations

import numpy as np


def sweep(scores: np.ndarray, labels: np.ndarray, n_points: int = 101) -> dict:
    """labels: 1 if the strong side won the human battle.

    Thresholds are score quantiles so each point routes a known fraction of
    traffic to the strong model (cost is controlled directly, like RouteLLM's
    calibrated thresholds).
    """
    qs = np.linspace(0, 1, n_points)
    cost, quality = [], []
    for q in qs:
        # Route the top-(1-q) fraction of scores to strong.
        if q == 0:
            routed_strong = np.ones_like(scores, dtype=bool)
        else:
            routed_strong = scores > np.quantile(scores, q)
        correct = np.where(routed_strong, labels == 1, labels == 0)
        cost.append(float(routed_strong.mean()))
        quality.append(float(correct.mean()))
    return {"cost": cost, "quality": quality}


def pgr_curve(curve: dict, labels: np.ndarray) -> dict:
    """Normalize quality between always-weak and always-strong anchors."""
    q_strong = float((labels == 1).mean())  # always-strong quality
    q_weak = float((labels == 0).mean())  # always-weak quality
    gap = q_strong - q_weak
    pgr = [(q - q_weak) / gap for q in curve["quality"]]
    return {"cost": curve["cost"], "pgr": pgr, "q_strong": q_strong, "q_weak": q_weak}


def apgr(pgr: dict) -> float:
    """Area under PGR vs cost, via trapezoid on the sorted cost axis."""
    order = np.argsort(pgr["cost"])
    x = np.array(pgr["cost"])[order]
    y = np.array(pgr["pgr"])[order]
    return float(np.trapezoid(y, x))


def cpt(pgr: dict, target: float) -> float | None:
    """Min fraction of strong-model calls reaching `target` PGR; None if never."""
    pts = sorted(zip(pgr["cost"], pgr["pgr"]))
    for c, p in pts:
        if p >= target:
            return float(c)
    return None


def ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of P(strong wins) predictions."""
    bins = np.clip((scores * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(scores[mask].mean() - (labels[mask] == 1).mean())
    return float(total)
