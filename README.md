# tiered-model-router

Training a small router that predicts, per query, whether a cheap LLM can handle it or it needs an expensive one — then routing accordingly. Goal: frontier-quality answers at a fraction of the cost, with calibration and cost/quality tradeoff curves to prove it (or honest curves showing where it fails).

**Status: work in progress.** v0 is trained and evaluated; its headline result is a *negative* one, documented below, that motivates v1. Every number in this README comes from a run in this repo — no invented metrics.

## Formulation

RouteLLM-style binary win-rate prediction ([Ong et al., ICLR 2025](https://arxiv.org/abs/2406.18665)): learn `P(strong model wins | query)` from preference data, route to the weak model when the predicted win probability is below a threshold. Sweeping the threshold traces the whole cost/quality curve from one trained model. Full design-space research (RouteLLM, RouterBench, FrugalGPT/AutoMix cascades, Not Diamond, Weave/Avengers-Pro) with links: [docs/phase1-research.md](docs/phase1-research.md).

## Data

[lmarena-ai/arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k): 57,477 raw battles over 64 models. We fit Bradley-Terry scores on all decisive battles, cluster models into 10 tiers (1-D k-means over BT scores), take tiers 0–1 as the strong class and tier 2 as weak, and keep decisive cross-class battles:

| pipeline stage | count |
|---|---|
| raw battles | 57,477 |
| ties (dropped) | 17,761 |
| decisive cross-class battles kept | 7,849 (13.7% of raw) |
| train / val / test | 6,279 / 785 / 785 |
| P(strong side wins) | 0.656 |

That last number is why routing is possible at all: the strong class only beats the weak class about 2 in 3 times, so roughly a third of queries don't need the expensive model.

## v0: frozen embeddings + logistic regression — an honest negative result

v0 encodes prompts with `all-MiniLM-L6-v2` and fits a class-weighted logistic regression. **It learns almost nothing** (val AUC 0.506). Test-split routing metrics (APGR = area under the PGR/cost curve; higher is better, random ≈ 0.5 by construction):

| router | APGR | CPT@50% | CPT@80% | ECE |
|---|---|---|---|---|
| v0 embed+LR | 0.553 | 0.41 | 0.74 | 0.154 |
| random | 0.548 | 0.44 | 0.77 | — |
| prompt-length heuristic | 0.530 | 0.52 | 0.74 | — |

Diagnosis (`scripts/diagnose_v0.py`, all reproducible):

- Probe tasks on the same embeddings (predict has-code / prompt-length) hit **AUC 0.93** → pipeline and features are fine.
- Train AUC on real labels (0.652) ≈ train AUC on shuffled labels (0.625) → the fit is mostly memorization.
- 5-fold CV AUC 0.529 ± 0.013 vs 0.489 shuffled → a real but tiny signal.
- Restricting to a clean fixed pair (GPT-4-family vs Mixtral/GPT-3.5/Llama-70b) doesn't rescue it: CV AUC 0.517 on 1,682 battles.

**Conclusion: a single human vote on a single stochastic generation is a very noisy label.** Prompt-only difficulty signal exists in raw Arena preferences but is too weak for embeddings+LR at this scale. This reproduces, from the ground up, why RouteLLM's strong results depended on data augmentation with verifiable ("golden") labels rather than raw preferences alone.

## Roadmap

- **v1 (next): correctness-based labels.** Train against "did the cheap model actually get it right" using verifiable tasks — starting from [RouterBench](https://arxiv.org/abs/2403.12031)'s 405k precomputed inference outcomes ($0 in API spend), then a small self-generated golden set for the exact model pair we deploy.
- Fine-tuned small encoder router; cluster-scorer (Avengers-Pro/Weave-style) baseline.
- Phase 3: live dispatch via OpenRouter, end-to-end cost/quality/latency vs always-frontier, demo on Railway.

## Reproduce

```bash
uv sync
uv run pytest                          # unit tests (label mapping, BT tiering)
uv run python scripts/prepare_data.py  # download 55k battles -> labeled splits
uv run python scripts/train_v0.py      # embeddings + LR -> artifacts/v0/
uv run python scripts/eval_v0.py       # eval table + curves -> artifacts/v0/
uv run python scripts/diagnose_v0.py   # the negative-result forensics
```

Runs entirely locally on an M-series Mac (no API keys needed so far).

## Limitations & failure modes so far

- Arena preference labels are noisy and 2024-era; tier assignment is derived from this dataset's battles, not the global leaderboard — our "strong" class includes models a 2026 reader wouldn't call strong.
- Dropping ties discards 31% of the data, including exactly the "both fine" cases a router would love to send cheap; v1's correctness labels fix this.
- The offline quality proxy ("did the routed side win the human battle") inherits all the noise above; it can rank routers but the absolute numbers shouldn't be quoted as product claims.
