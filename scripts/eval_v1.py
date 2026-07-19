"""Evaluate router v1 on the held-out test split, with honest baselines.

Quality is direct now: a policy's quality = fraction of test prompts where
the model it routed to answered correctly. Cost is reported two ways:
fraction of traffic sent to the strong model, and measured dollars per
prompt (from the actual per-request costs logged during label generation).

Outputs artifacts/v1/eval.json + curves.png. Every README number comes from
this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from router.embed import embed_split
from router.evaluate import ece

ROOT = Path(__file__).resolve().parent.parent
DATA, ART = ROOT / "data", ROOT / "artifacts" / "v1"
RNG = np.random.default_rng(0)
CHEAP_SLUG, STRONG_SLUG = "claude-haiku-4-5", "claude-opus-4-8"


def load_costs(slug: str) -> dict[str, float]:
    path = DATA / f"responses_{slug}.jsonl"
    with open(path) as f:
        return {r["id"]: r["cost_usd"] for line in f if (r := json.loads(line))}


def policy_metrics(route_strong: np.ndarray, test: pd.DataFrame) -> dict:
    quality = np.where(route_strong, test["strong_correct"], test["cheap_correct"]).mean()
    dollars = np.where(route_strong, test["cost_strong"], test["cost_cheap"]).mean()
    return {"frac_strong": float(route_strong.mean()), "quality": float(quality),
            "usd_per_prompt": float(dollars)}


def sweep(scores_cheap_ok: np.ndarray, test: pd.DataFrame, n: int = 101) -> list[dict]:
    """Route to strong the fraction q of prompts with LOWEST P(cheap correct)."""
    pts = []
    for q in np.linspace(0, 1, n):
        if q == 0:
            route_strong = np.zeros(len(test), dtype=bool)
        else:
            route_strong = scores_cheap_ok <= np.quantile(scores_cheap_ok, q)
        pts.append(policy_metrics(route_strong, test))
    return pts


def pgr(points: list[dict], q_weak: float, q_strong: float) -> list[dict]:
    gap = q_strong - q_weak
    return [{**p, "pgr": (p["quality"] - q_weak) / gap if gap > 0 else 0.0} for p in points]


def apgr(points: list[dict]) -> float:
    pts = sorted(points, key=lambda p: p["frac_strong"])
    x = np.array([p["frac_strong"] for p in pts])
    y = np.array([p["pgr"] for p in pts])
    return float(np.trapezoid(y, x))


def cpt(points: list[dict], target: float) -> float | None:
    for p in sorted(points, key=lambda p: p["frac_strong"]):
        if p["pgr"] >= target:
            return round(p["frac_strong"], 3)
    return None


def main() -> None:
    test = pd.read_parquet(DATA / "v1_split_test.parquet")
    cheap_costs, strong_costs = load_costs(CHEAP_SLUG), load_costs(STRONG_SLUG)
    test = test.assign(
        cost_cheap=test["id"].map(cheap_costs),
        cost_strong=test["id"].map(strong_costs),
    )
    X_test = embed_split(test, DATA, "v1_test")
    clf = joblib.load(ART / "model.joblib")

    q_weak = float(test["cheap_correct"].mean())
    q_strong = float(test["strong_correct"].mean())

    scores = {
        "v1_embed_lr": clf.predict_proba(X_test)[:, 1],
        "random": RNG.random(len(test)),
        # Longer prompt => assume harder => lower "cheap ok" score.
        "length_heuristic": 1.0 - (
            test["prompt_text"].str.len().to_numpy(dtype=float)
            / test["prompt_text"].str.len().max()
        ),
    }

    results: dict = {
        "n_test": len(test),
        "anchors": {
            "always_cheap": policy_metrics(np.zeros(len(test), dtype=bool), test),
            "always_strong": policy_metrics(np.ones(len(test), dtype=bool), test),
        },
    }

    plt.figure(figsize=(7.5, 5))
    for name, s in scores.items():
        pts = pgr(sweep(np.asarray(s, dtype=float), test), q_weak, q_strong)
        results[name] = {
            "apgr": round(apgr(pts), 4),
            "cpt_50": cpt(pts, 0.5),
            "cpt_80": cpt(pts, 0.8),
            "ece": round(ece(np.asarray(s), test["cheap_correct"].to_numpy()), 4)
            if name == "v1_embed_lr" else None,
        }
        plt.plot([p["frac_strong"] for p in pts], [p["pgr"] for p in pts],
                 label=f"{name} (APGR {results[name]['apgr']:.3f})")

    # Task-type rule baseline (uses tier metadata a deployed router wouldn't have;
    # included as the "hand-written MAIN/MINI heuristic" reference point).
    rule = test["source"].isin(["math-l4", "math-l5", "mmlu-pro"]).to_numpy()
    rule_m = policy_metrics(rule, test)
    rule_m["pgr"] = (rule_m["quality"] - q_weak) / (q_strong - q_weak)
    results["tier_rule_baseline"] = {k: round(v, 4) for k, v in rule_m.items()}
    plt.scatter([rule_m["frac_strong"]], [rule_m["pgr"]], marker="x", s=80, c="k",
                label="hand tier-rule", zorder=5)

    plt.xlabel("cost (fraction routed to Opus 4.8)")
    plt.ylabel("PGR (fraction of Haiku→Opus quality gap recovered)")
    plt.title(f"Router v1 — correctness labels (test n={len(test)})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ART / "curves.png", dpi=150)

    with open(ART / "eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
