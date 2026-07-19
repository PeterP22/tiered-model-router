"""Async Anthropic API runner with cost tracking.

Same interface as OpenRouterClient.complete(): returns (text, usage, cost_usd)
so scripts/generate_labels.py can use either provider. Prices are per-MTok
from the current Anthropic price list; update PRICES if they change.

Note: no temperature/top_p — Opus 4.8-family models reject sampling params.
Thinking is left at each model's default (off for Opus 4.8 when omitted) so
both tiers answer under comparable settings; revisit if we study
thinking-on routing later.
"""

from __future__ import annotations

import asyncio

from anthropic import AsyncAnthropic

PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),  # intro pricing through 2026-08-31
    "claude-opus-4-8": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
}


class AnthropicRunner:
    def __init__(self, concurrency: int = 8):
        self._client = AsyncAnthropic(max_retries=4)
        self._sem = asyncio.Semaphore(concurrency)
        self.total_cost = 0.0

    async def complete(self, model: str, prompt: str, max_tokens: int = 512) -> tuple[str, dict, float]:
        async with self._sem:
            msg = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = {
            "prompt_tokens": msg.usage.input_tokens,
            "completion_tokens": msg.usage.output_tokens,
        }
        pin, pout = PRICES.get(model, (0.0, 0.0))
        cost = (usage["prompt_tokens"] * pin + usage["completion_tokens"] * pout) / 1e6
        self.total_cost += cost
        return text, usage, cost
