"""Build the main labeling set, weighted toward the measured capability gap.

Mix (450 total): gsm8k 40, mmlu 60, math-l3 50, math-l4 100, math-l5 100,
mmlu-pro 100 — heavy on the tiers where the probe found a real Haiku->Opus
gap (docs: probe results in README). Uses the same seed-1 shuffles as
prepare_probe_prompts.py and skips each tier's first 20 rows so no prompt
appears in both sets. Output: data/main_prompts.parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA = Path(__file__).resolve().parent.parent / "data"
SKIP = 20  # rows consumed by the probe set per tier

GSM8K_SUFFIX = "\n\nSolve step by step, then give the final line as 'Answer: <number>'."
MMLU_SUFFIX = "\n\nThink briefly, then give the final line as 'Answer: <letter>'."
MATH_SUFFIX = ("\n\nSolve step by step, then give the final line as "
               "'Answer: <answer>' using the same format as the problem (LaTeX if needed).")

LETTERS = [chr(ord("A") + i) for i in range(10)]
MIX = {"gsm8k": 40, "mmlu": 60, "math-l3": 50, "math-l4": 100, "math-l5": 100, "mmlu-pro": 100}


def main() -> None:
    rows = []

    gsm = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=1)
    for i, ex in enumerate(gsm.select(range(SKIP, SKIP + MIX["gsm8k"]))):
        rows.append({"id": f"main-gsm8k-{i}", "source": "gsm8k",
                     "prompt": ex["question"] + GSM8K_SUFFIX,
                     "gold": ex["answer"].split("####")[-1].strip()})

    mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=1)
    for i, ex in enumerate(mmlu.select(range(SKIP, SKIP + MIX["mmlu"]))):
        q = ex["question"] + "\n" + "\n".join(
            f"{letter}. {c}" for letter, c in zip(LETTERS, ex["choices"]))
        rows.append({"id": f"main-mmlu-{i}", "source": "mmlu",
                     "prompt": q + MMLU_SUFFIX, "gold": LETTERS[ex["answer"]]})

    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for level in (3, 4, 5):
        want = MIX[f"math-l{level}"]
        subset = math.filter(lambda ex: ex["level"] == level).shuffle(seed=1)
        avail = len(subset) - SKIP
        take = min(want, avail)
        if take < want:
            print(f"note: math-l{level} has only {avail} unused rows (wanted {want})")
        for i, ex in enumerate(subset.select(range(SKIP, SKIP + take))):
            rows.append({"id": f"main-math-l{level}-{i}", "source": f"math-l{level}",
                         "prompt": ex["problem"] + MATH_SUFFIX, "gold": ex["answer"]})

    pro = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=1)
    for i, ex in enumerate(pro.select(range(SKIP, SKIP + MIX["mmlu-pro"]))):
        q = ex["question"] + "\n" + "\n".join(
            f"{letter}. {c}" for letter, c in zip(LETTERS, ex["options"]))
        rows.append({"id": f"main-mmlupro-{i}", "source": "mmlu-pro",
                     "prompt": q + MMLU_SUFFIX, "gold": ex["answer"]})

    df = pd.DataFrame(rows)
    df.to_parquet(DATA / "main_prompts.parquet", index=False)
    print(df.groupby("source").size())
    print(f"total: {len(df)} prompts -> data/main_prompts.parquet")


if __name__ == "__main__":
    main()
