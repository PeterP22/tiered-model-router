"""Download Arena preference data and produce labeled router training splits.

Pipeline (RouteLLM recipe, documented in docs/phase1-research.md):
  1. Download lmarena-ai/arena-human-preference-55k.
  2. Fit Bradley-Terry scores on all decisive battles; cluster into 10 tiers.
  3. Strong class = tiers 0-1, weak class = tier 2.
  4. Label cross-class battles: 1 = strong side won, 0 = weak side won.
  5. Stratified 80/10/10 train/val/test split (split by battle; prompts in the
     55k set are unique per battle).

Outputs: data/battles_labeled.parquet, data/split_{train,val,test}.parquet,
data/tiers.json. Every drop count is printed — these numbers go in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from router.elo import assign_tiers, fit_bt_scores
from router.labels import label_battle

DATA = Path(__file__).resolve().parent.parent / "data"
STRONG_TIERS = {0, 1}
WEAK_TIERS = {2}


def extract_prompt(raw: str) -> str:
    """The 55k dataset stores prompt as a JSON-encoded list of turns; join them."""
    try:
        turns = json.loads(raw)
        if isinstance(turns, list):
            return "\n".join(str(t) for t in turns if t)
    except (json.JSONDecodeError, TypeError):
        pass
    return str(raw)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    ds = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
    df = ds.to_pandas()
    print(f"raw battles: {len(df)}")

    # Dataset encodes the outcome as one-hot winner_model_a/b/tie columns.
    def winner(row) -> str:
        if row["winner_model_a"] == 1:
            return "model_a"
        if row["winner_model_b"] == 1:
            return "model_b"
        return "tie"

    df["winner"] = df.apply(winner, axis=1)
    df["prompt_text"] = df["prompt"].map(extract_prompt)

    scores = fit_bt_scores(df[["model_a", "model_b", "winner"]])
    tiers = assign_tiers(scores, n_tiers=10)
    tier_table = (
        pd.DataFrame({"model": list(scores), "bt_score": list(scores.values())})
        .assign(tier=lambda d: d["model"].map(tiers))
        .sort_values("bt_score", ascending=False)
    )
    print(f"models: {len(tier_table)}")
    print("strong class:", sorted(m for m, t in tiers.items() if t in STRONG_TIERS))
    print("weak class:", sorted(m for m, t in tiers.items() if t in WEAK_TIERS))

    df["label"] = [
        label_battle(a, b, w, tiers, STRONG_TIERS, WEAK_TIERS)
        for a, b, w in zip(df["model_a"], df["model_b"], df["winner"])
    ]
    n_tie = int((df["winner"] == "tie").sum())
    labeled = df.dropna(subset=["label"]).copy()
    labeled["label"] = labeled["label"].astype(int)
    print(f"ties in raw data (dropped): {n_tie}")
    print(f"cross-class decisive battles kept: {len(labeled)} "
          f"({len(labeled) / len(df):.1%} of raw)")
    print(f"label balance P(strong wins): {labeled['label'].mean():.3f}")

    keep = labeled[["id", "prompt_text", "model_a", "model_b", "winner", "label"]]
    keep.to_parquet(DATA / "battles_labeled.parquet", index=False)

    train, rest = train_test_split(keep, test_size=0.2, random_state=0, stratify=keep["label"])
    val, test = train_test_split(rest, test_size=0.5, random_state=0, stratify=rest["label"])
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(DATA / f"split_{name}.parquet", index=False)
        print(f"{name}: {len(part)} rows")

    with open(DATA / "tiers.json", "w") as f:
        json.dump(
            {
                "bt_scores": {m: float(s) for m, s in scores.items()},
                "tiers": tiers,
                "strong_tiers": sorted(STRONG_TIERS),
                "weak_tiers": sorted(WEAK_TIERS),
            },
            f,
            indent=2,
        )
    print(f"wrote {DATA}/tiers.json")


if __name__ == "__main__":
    main()
