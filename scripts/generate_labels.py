"""Run golden prompts through cheap + strong models and grade the answers.

Default provider is the Anthropic API (ANTHROPIC_API_KEY in .env or env);
--provider openrouter switches to OpenRouter (OPENROUTER_API_KEY).

Resumable: responses append to data/responses_<model-slug>.jsonl keyed by
prompt id; rerunning skips completed ids. Running cost prints every 25
completions and at exit — no invisible spend. A --max-cost guard aborts new
requests once the session spend crosses it.

Usage:
  uv run python scripts/generate_labels.py \
      --cheap claude-haiku-4-5 --strong claude-opus-4-8 --limit 10   # smoke
  uv run python scripts/generate_labels.py \
      --cheap claude-haiku-4-5 --strong claude-opus-4-8 --max-cost 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from router.grade import grade_gsm8k, grade_mmlu

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "-")


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["id"]: r for line in f if (r := json.loads(line))}


class CostCapReached(Exception):
    pass


async def run_model(model: str, prompts: pd.DataFrame, runner, max_cost: float | None) -> None:
    out_path = DATA / f"responses_{slug(model)}.jsonl"
    done = load_done(out_path)
    todo = prompts[~prompts["id"].isin(done)]
    print(f"[{model}] {len(done)} cached, {len(todo)} to run")
    if todo.empty:
        return

    lock = asyncio.Lock()
    counter = {"n": 0}

    async def one(row) -> None:
        if max_cost is not None and runner.total_cost >= max_cost:
            raise CostCapReached
        text, usage, cost = await runner.complete(model, row.prompt)
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
                      f"session cost so far: ${runner.total_cost:.4f}")

    results = await asyncio.gather(*(one(row) for row in todo.itertuples()),
                                   return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception) and not isinstance(r, CostCapReached)]
    if any(isinstance(r, CostCapReached) for r in results):
        print(f"[{model}] STOPPED: --max-cost ${max_cost} reached "
              f"(spent ${runner.total_cost:.4f}); rerun to resume later")
    if errors:
        print(f"[{model}] {len(errors)} requests failed (rerun to retry); first: {errors[0]!r}")


async def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", default="claude-haiku-4-5")
    ap.add_argument("--strong", default="claude-opus-4-8")
    ap.add_argument("--provider", choices=["anthropic", "openrouter"], default="anthropic")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N prompts")
    ap.add_argument("--max-cost", type=float, default=None, help="hard USD cap for this session")
    args = ap.parse_args()

    prompts = pd.read_parquet(DATA / "golden_prompts.parquet")
    if args.limit:
        # Take from every source so a smoke test exercises all graders.
        prompts = pd.concat(
            [g.head(args.limit // 2) for _, g in prompts.groupby("source")]
        )
    print(f"prompts: {len(prompts)} | provider: {args.provider} | "
          f"cheap: {args.cheap} | strong: {args.strong}")

    if args.provider == "anthropic":
        from router.anthropic_client import AnthropicRunner

        runner = AnthropicRunner(concurrency=8)
        for model in (args.cheap, args.strong):
            await run_model(model, prompts, runner, args.max_cost)
    else:
        import httpx

        from router.openrouter import OpenRouterClient

        or_client = OpenRouterClient(concurrency=8)
        async with httpx.AsyncClient() as http:
            await or_client.load_pricing(http)

            class _Runner:
                total_cost = 0.0

                async def complete(self, model, prompt, max_tokens=512):
                    out = await or_client.complete(http, model, prompt, max_tokens)
                    self.total_cost = or_client.total_cost
                    return out

            runner = _Runner()
            for model in (args.cheap, args.strong):
                await run_model(model, prompts, runner, args.max_cost)

    # Merge into labels: needs_strong = cheap wrong AND strong right.
    cheap = load_done(DATA / f"responses_{slug(args.cheap)}.jsonl")
    strong = load_done(DATA / f"responses_{slug(args.strong)}.jsonl")
    rows = [
        {
            "id": pid,
            "cheap_correct": cheap[pid]["correct"],
            "strong_correct": strong[pid]["correct"],
            "needs_strong": (not cheap[pid]["correct"]) and strong[pid]["correct"],
        }
        for pid in prompts["id"]
        if pid in cheap and pid in strong
    ]
    if not rows:
        print("\nno completed prompt pairs yet — nothing to merge")
        print(f"TOTAL SESSION COST: ${runner.total_cost:.4f}")
        return
    labels = pd.DataFrame(rows).merge(prompts, on="id")
    labels.to_parquet(DATA / "golden_labels.parquet", index=False)

    print(f"\nlabels written: {len(labels)}")
    print(f"cheap  ({args.cheap}) accuracy: {labels['cheap_correct'].mean():.3f}")
    print(f"strong ({args.strong}) accuracy: {labels['strong_correct'].mean():.3f}")
    print(f"needs_strong rate: {labels['needs_strong'].mean():.3f}")
    print(f"TOTAL SESSION COST: ${runner.total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
