"""Map raw Arena battles to binary router labels.

A battle is usable only when one side is in the strong class and the other in
the weak class; the label is whether the strong side won. Ties and battles
inside a single class carry no signal about strong-vs-weak and are dropped
(returned as None) — the counts of dropped rows are reported by
scripts/prepare_data.py so the filtering stays visible.
"""

from __future__ import annotations


def label_battle(
    model_a: str,
    model_b: str,
    winner: str,
    tiers: dict[str, int],
    strong_tiers: set[int],
    weak_tiers: set[int],
) -> int | None:
    ta, tb = tiers.get(model_a), tiers.get(model_b)
    if ta is None or tb is None:
        return None

    def cls(t: int) -> str | None:
        if t in strong_tiers:
            return "strong"
        if t in weak_tiers:
            return "weak"
        return None

    ca, cb = cls(ta), cls(tb)
    if ca is None or cb is None or ca == cb:
        return None
    if winner not in ("model_a", "model_b"):
        return None  # ties ("tie", "tie (bothbad)") carry no strong-vs-weak signal

    winner_class = ca if winner == "model_a" else cb
    return 1 if winner_class == "strong" else 0
