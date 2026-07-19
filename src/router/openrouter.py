"""Minimal async OpenRouter client with live-pricing cost tracking.

Reads OPENROUTER_API_KEY from the environment (a project-root .env is loaded
by the scripts). Every completion returns (text, usage, cost_usd) where cost
is computed from OpenRouter's public pricing endpoint at client start — the
running total is printed by callers so no spend is ever invisible.
"""

from __future__ import annotations

import asyncio
import os

import httpx

API = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    def __init__(self, concurrency: int = 8, max_retries: int = 4):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set (put it in .env or export it)")
        self._headers = {"Authorization": f"Bearer {key}"}
        self._sem = asyncio.Semaphore(concurrency)
        self._max_retries = max_retries
        self._pricing: dict[str, tuple[float, float]] = {}
        self.total_cost = 0.0

    async def load_pricing(self, client: httpx.AsyncClient) -> None:
        r = await client.get(f"{API}/models")
        r.raise_for_status()
        for m in r.json()["data"]:
            p = m.get("pricing", {})
            self._pricing[m["id"]] = (float(p.get("prompt", 0)), float(p.get("completion", 0)))

    def cost_of(self, model: str, usage: dict) -> float:
        pin, pout = self._pricing.get(model, (0.0, 0.0))
        return usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout

    async def complete(
        self,
        client: httpx.AsyncClient,
        model: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> tuple[str, dict, float]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        async with self._sem:
            for attempt in range(self._max_retries):
                try:
                    r = await client.post(f"{API}/chat/completions", json=payload,
                                          headers=self._headers, timeout=120)
                    if r.status_code == 429 or r.status_code >= 500:
                        raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                    r.raise_for_status()
                    body = r.json()
                    text = body["choices"][0]["message"]["content"] or ""
                    usage = body.get("usage", {})
                    cost = self.cost_of(model, usage)
                    self.total_cost += cost
                    return text, usage, cost
                except (httpx.HTTPStatusError, httpx.TransportError):
                    if attempt == self._max_retries - 1:
                        raise
                    await asyncio.sleep(2**attempt)
        raise RuntimeError("unreachable")
