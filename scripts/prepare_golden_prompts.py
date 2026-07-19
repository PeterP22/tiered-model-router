"""Sample verifiable prompts for golden-label generation.

GSM8K (grade-school math, exact numeric answers) + MMLU (4-choice knowledge,
letter answers). Output: data/golden_prompts.parquet with columns
id, source, prompt, gold. Sizes are CLI-controlled so spend stays deliberate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA = Path(__file__).resolve().parent.parent / "data"

GSM8K_SUFFIX = "\n\nSolve step by step, then give the final line as 'Answer: <number>'."
MMLU_SUFFIX = "\n\nThink briefly, then give the final line as 'Answer: <letter>'."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gsm8k", type=int, default=2000)
    ap.add_argument("--n-mmlu", type=int, default=2000)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    rows = []
    gsm = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
    for i, ex in enumerate(gsm.select(range(args.n_gsm8k))):
        gold = ex["answer"].split("####")[-1].strip()
        rows.append({"id": f"gsm8k-{i}", "source": "gsm8k",
                     "prompt": ex["question"] + GSM8K_SUFFIX, "gold": gold})

    mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0)
    for i, ex in enumerate(mmlu.select(range(args.n_mmlu))):
        letters = ["A", "B", "C", "D"]
        q = ex["question"] + "\n" + "\n".join(
            f"{letter}. {c}" for letter, c in zip(letters, ex["choices"])
        )
        rows.append({"id": f"mmlu-{i}", "source": "mmlu",
                     "prompt": q + MMLU_SUFFIX, "gold": letters[ex["answer"]]})

    df = pd.DataFrame(rows)
    df.to_parquet(DATA / "golden_prompts.parquet", index=False)
    print(f"wrote {len(df)} prompts ({args.n_gsm8k} gsm8k + {args.n_mmlu} mmlu)")
    print(f"mean prompt chars: {df['prompt'].str.len().mean():.0f}")


if __name__ == "__main__":
    main()
