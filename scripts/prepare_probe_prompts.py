"""Build the difficulty-probe prompt set: 20 prompts per tier, 6 tiers.

Tiers span easy -> hard so we can map where the Haiku->Opus capability
boundary sits in 2026: gsm8k, mmlu (easy anchors), math-l3/l4/l5
(HuggingFaceH4/MATH-500 by level), mmlu-pro (TIGER-Lab/MMLU-Pro).
Output: data/probe_prompts.parquet (same schema as golden_prompts.parquet).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA = Path(__file__).resolve().parent.parent / "data"

GSM8K_SUFFIX = "\n\nSolve step by step, then give the final line as 'Answer: <number>'."
MMLU_SUFFIX = "\n\nThink briefly, then give the final line as 'Answer: <letter>'."
MATH_SUFFIX = ("\n\nSolve step by step, then give the final line as "
               "'Answer: <answer>' using the same format as the problem (LaTeX if needed).")

LETTERS = [chr(ord("A") + i) for i in range(10)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=20)
    args = ap.parse_args()
    n = args.per_tier
    rows = []

    gsm = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=1)
    for i, ex in enumerate(gsm.select(range(n))):
        rows.append({"id": f"probe-gsm8k-{i}", "source": "gsm8k",
                     "prompt": ex["question"] + GSM8K_SUFFIX,
                     "gold": ex["answer"].split("####")[-1].strip()})

    mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=1)
    for i, ex in enumerate(mmlu.select(range(n))):
        q = ex["question"] + "\n" + "\n".join(
            f"{letter}. {c}" for letter, c in zip(LETTERS, ex["choices"]))
        rows.append({"id": f"probe-mmlu-{i}", "source": "mmlu",
                     "prompt": q + MMLU_SUFFIX, "gold": LETTERS[ex["answer"]]})

    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for level in (3, 4, 5):
        subset = math.filter(lambda ex: ex["level"] == level).shuffle(seed=1)
        for i, ex in enumerate(subset.select(range(min(n, len(subset))))):
            rows.append({"id": f"probe-math-l{level}-{i}", "source": f"math-l{level}",
                         "prompt": ex["problem"] + MATH_SUFFIX, "gold": ex["answer"]})

    pro = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=1)
    for i, ex in enumerate(pro.select(range(n))):
        q = ex["question"] + "\n" + "\n".join(
            f"{letter}. {c}" for letter, c in zip(LETTERS, ex["options"]))
        rows.append({"id": f"probe-mmlupro-{i}", "source": "mmlu-pro",
                     "prompt": q + MMLU_SUFFIX, "gold": ex["answer"]})

    df = pd.DataFrame(rows)
    DATA.mkdir(exist_ok=True)
    df.to_parquet(DATA / "probe_prompts.parquet", index=False)
    print(df.groupby("source").size())
    print(f"total: {len(df)} prompts -> data/probe_prompts.parquet")


if __name__ == "__main__":
    main()
