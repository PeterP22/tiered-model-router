"""Run golden prompts through cheap + strong models on OpenRouter and grade.

Resumable: responses append to data/responses_<model-slug>.jsonl keyed by
prompt id; rerunning skips completed ids. Running cost is printed every 25
completions and at exit — no invisible spend.

Usage:
  uv run python scripts/generate_labels.py --cheap anthropic/claude-haiku-4.5 \
      --strong anthropic/claude-opus-4.8 --limit 20        # smoke test
  uv run python scripts/generate_labels.py --cheap ... --strong ...  # full run
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from router.grade import grade_gsm8k, grade_mmlu
from router.openrouter import OpenRouterClient

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "-")


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["id"]: r for line in f if (r := json.loads(line))}


async def run_model(model: str, prompts: pd.DataFrame, or_client: OpenRouterClient,
                    http: httpx.AsyncClient) -> None:
    out_path = DATA / f"responses_{slug(model)}.jsonl"
    done = load_done(out_path)
    todo = prompts[~prompts["id"].isin(done)]
    print(f"[{model}] {len(done)} cached, {len(todo)} to run")
    if todo.empty:
        return

    lock = asyncio.Lock()
    counter = {"n": 0}

    async def one(row) -> None:
        text, usage, cost = await or_client.complete(http, model, row.prompt)
        grader = grade_gsm8k if row.source == "gsm8k" else grade_mmlu
        rec = {
            "id": row.id, "model": model, "correct": bool(grader(text, row.gold)),
            "output": text, "usage": usage, "cost_usd": cost,
        }
        async with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                print(f"[{model}] {counter['n']}/{len(todo)} done, "
                      f"session cost so far: ${or_client.total_cost:.4f}")

    await asyncio.gather(*(one(row) for row in todo.itertuples()))


async def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--strong", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on first N prompts")
    args = ap.parse_args()

    prompts = pd.read_parquet(DATA / "golden_prompts.parquet")
    if args.limit:
        # Take from every source so a smoke test exercises all graders.
        prompts = pd.concat(
            [g.head(args.limit // 2) for _, g in prompts.groupby("source")]
        )
    print(f"prompts: {len(prompts)}")

    or_client = OpenRouterClient(concurrency=8)
    async with httpx.AsyncClient() as http:
        await or_client.load_pricing(http)
        for model in (args.cheap, args.strong):
            await run_model(model, prompts, or_client, http)

    # Merge into labels: needs_strong = cheap wrong AND strong right.
    cheap = load_done(DATA / f"responses_{slug(args.cheap)}.jsonl")
    strong = load_done(DATA / f"responses_{slug(args.strong)}.jsonl")
    rows = []
    for pid in prompts["id"]:
        if pid in cheap and pid in strong:
            rows.append({
                "id": pid,
                "cheap_correct": cheap[pid]["correct"],
                "strong_correct": strong[pid]["correct"],
                "needs_strong": (not cheap[pid]["correct"]) and strong[pid]["correct"],
            })
    labels = pd.DataFrame(rows).merge(prompts, on="id")
    labels.to_parquet(DATA / "golden_labels.parquet", index=False)

    print(f"\nlabels written: {len(labels)}")
    print(f"cheap  ({args.cheap}) accuracy: {labels['cheap_correct'].mean():.3f}")
    print(f"strong ({args.strong}) accuracy: {labels['strong_correct'].mean():.3f}")
    print(f"needs_strong rate: {labels['needs_strong'].mean():.3f}")
    print(f"TOTAL SESSION COST: ${or_client.total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
