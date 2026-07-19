import pandas as pd

from router.elo import fit_bt_scores, assign_tiers


def _battles():
    # A beats B mostly, B beats C mostly => score order A > B > C
    rows = []
    rows += [("A", "B", "model_a")] * 8 + [("A", "B", "model_b")] * 2
    rows += [("B", "C", "model_a")] * 8 + [("B", "C", "model_b")] * 2
    rows += [("A", "C", "model_a")] * 9 + [("A", "C", "model_b")] * 1
    return pd.DataFrame(rows, columns=["model_a", "model_b", "winner"])


def test_bt_score_ordering():
    scores = fit_bt_scores(_battles())
    assert scores["A"] > scores["B"] > scores["C"]


def test_assign_tiers_orders_by_score():
    scores = {"A": 3.0, "B": 2.9, "C": 0.1, "D": 0.0}
    tiers = assign_tiers(scores, n_tiers=2)
    assert tiers["A"] == 0 and tiers["B"] == 0
    assert tiers["C"] == 1 and tiers["D"] == 1
