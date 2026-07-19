"""Evaluate router v0 against baselines on the held-out test split.

Writes artifacts/v0/eval.json and artifacts/v0/curves.png. Every number in the
README's eval table comes from this script's output.
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
from router.evaluate import apgr, cpt, ece, pgr_curve, sweep

ROOT = Path(__file__).resolve().parent.parent
DATA, ART = ROOT / "data", ROOT / "artifacts" / "v0"
RNG = np.random.default_rng(0)


def main() -> None:
    test = pd.read_parquet(DATA / "split_test.parquet")
    labels = test["label"].to_numpy()
    X_test = embed_split(test, DATA, "test")

    clf = joblib.load(ART / "model.joblib")
    routers = {
        "v0_embed_lr": clf.predict_proba(X_test)[:, 1],
        "random": RNG.random(len(test)),
        "length_heuristic": (
            test["prompt_text"].str.len().to_numpy(dtype=float)
            / test["prompt_text"].str.len().max()
        ),
    }

    results: dict = {"n_test": len(test)}
    plt.figure(figsize=(7, 5))
    for name, scores in routers.items():
        pgr = pgr_curve(sweep(scores, labels), labels)
        results[name] = {
            "apgr": apgr(pgr),
            "cpt_50": cpt(pgr, 0.5),
            "cpt_80": cpt(pgr, 0.8),
            "ece": ece(scores, labels) if name == "v0_embed_lr" else None,
        }
        plt.plot(pgr["cost"], pgr["pgr"], label=f"{name} (APGR {results[name]['apgr']:.3f})")

    results["anchors"] = {
        "always_strong_quality": float((labels == 1).mean()),
        "always_weak_quality": float((labels == 0).mean()),
    }

    plt.xlabel("cost (fraction routed to strong model)")
    plt.ylabel("PGR (fraction of quality gap recovered)")
    plt.title("Router v0 — cost/quality tradeoff (test split, n=%d)" % len(test))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ART / "curves.png", dpi=150)

    with open(ART / "eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
