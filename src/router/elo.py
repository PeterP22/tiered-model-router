"""Bradley-Terry model scores and tier assignment from Arena battles.

fit_bt_scores fits the Bradley-Terry model as a logistic regression on
model-indicator difference vectors (the same estimator LMSYS uses for the
Arena leaderboard). assign_tiers groups models into n_tiers by 1-D k-means on
the BT scores — an approximation of RouteLLM's minimum-intra-tier-variance
dynamic program that is close enough for tier counts this small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def fit_bt_scores(battles: pd.DataFrame) -> dict[str, float]:
    """battles: columns model_a, model_b, winner ('model_a'|'model_b'). Ties excluded."""
    decisive = battles[battles["winner"].isin(["model_a", "model_b"])]
    models = sorted(set(decisive["model_a"]) | set(decisive["model_b"]))
    idx = {m: i for i, m in enumerate(models)}

    X = np.zeros((len(decisive), len(models)))
    rows = np.arange(len(decisive))
    X[rows, decisive["model_a"].map(idx)] = 1.0
    X[rows, decisive["model_b"].map(idx)] = -1.0
    y = (decisive["winner"] == "model_a").astype(int).to_numpy()

    # Near-unpenalized (C large); sklearn 1.8 deprecated penalty=None.
    lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=1000)
    lr.fit(X, y)
    return dict(zip(models, lr.coef_[0]))


def assign_tiers(scores: dict[str, float], n_tiers: int = 10) -> dict[str, int]:
    """Tier 0 = highest-scoring cluster. Requires len(scores) >= n_tiers."""
    models = list(scores)
    values = np.array([scores[m] for m in models]).reshape(-1, 1)
    km = KMeans(n_clusters=n_tiers, n_init=10, random_state=0).fit(values)
    # Relabel clusters so tier 0 has the highest mean score.
    order = np.argsort(-km.cluster_centers_.ravel())
    rank = {int(c): int(t) for t, c in enumerate(order)}
    return {m: rank[int(l)] for m, l in zip(models, km.labels_)}
