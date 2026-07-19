# Phase 1 — The Learned Model Router Design Space

*Research summary, July 2026. All numbers in this document are claims from the cited papers/vendors, not our results. Our own README will only ever contain numbers from runs we did.*

## TL;DR recommendation

Build a **RouteLLM-style binary win-rate router**: a small model that predicts `P(strong model wins | query)`, routed through a tunable threshold. Start with an **embedding + classifier head** (fast to train on M-series, honest baseline), then a **fine-tuned small encoder (BERT-class)** as the "real" router. Train on **Chatbot Arena preference data** (public on HuggingFace) with golden-label augmentation. Evaluate with **cost/quality tradeoff curves (PGR/APGR/CPT)** plus **calibration (reliability diagrams, ECE)**. Cascades and multi-way routing are documented below as the roads not taken — with reasons.

---

## 1. The landscape

### Academic anchors

| Paper | Year | One-liner |
|---|---|---|
| [FrugalGPT](https://arxiv.org/abs/2305.05176) (Chen, Zaharia, Zou — Stanford) | 2023 | The cascade OG: try cheap models in sequence, a learned scoring function decides if the answer is good enough or escalates. Claims GPT-4-level performance at up to 98% cost reduction. |
| [AutoMix](https://arxiv.org/abs/2310.12963) | 2023/24 | Cascade with **self-verification**: the small model few-shot-verifies its own answer; a POMDP-based meta-verifier decides escalation. |
| [RouterBench](https://arxiv.org/abs/2403.12031) (Martian) | 2024 | The eval framework: ~405k precomputed inference outcomes over 11 models and 8 tasks, so you can evaluate routers **without any API calls**. Defines the cost-quality plane, non-decreasing convex hull, and the **AIQ** metric (area under the routing curve). |
| [**RouteLLM**](https://arxiv.org/abs/2406.18665) (LMSYS/Berkeley, ICLR 2025) | 2024 | **The blueprint for this project.** Binary strong-vs-weak routing formulated as win-rate prediction from human preference data. Four router architectures; headline claim: >2x cost reduction at ~95% of GPT-4 quality on MT-Bench. [Code (Apache 2.0)](https://github.com/lm-sys/routellm) · [blog](https://www.lmsys.org/blog/2024-07-01-routellm/) |
| [Prompt-to-Leaderboard](https://arxiv.org/abs/2502.14855) (LMSYS) | 2025 | Per-prompt Elo: train a model to output a query-specific leaderboard. Router falls out as a byproduct — the generalization of win-rate routing to K models. |
| [Arch-Router](https://arxiv.org/abs/2506.16655) | 2025 | A 1.5B generative router that maps queries to human-defined **domain/action policies** rather than difficulty — "preference-aligned" routing, closer to intent routing than cost optimization. |
| [Dynamic Model Routing and Cascading survey](https://arxiv.org/abs/2603.04445) | 2026 | Organizes the field along three axes: **when** the decision happens (pre-inference routing vs mid/post-inference cascading), **what signals** feed it (query features, model metadata, past performance), and **how** it's computed (rules, classifiers, RL). Good map; read after RouteLLM. |

Also notable from 2025–2026 searches: [LLMRouterBench](https://arxiv.org/html/2601.07206v1) (larger unified routing benchmark), [R2-Router](https://arxiv.org/html/2602.02823v1) (reasoning-based routing), and a [growing awesome-list](https://github.com/MilkThink-Lab/Awesome-Routing-LLMs).

### Commercial state of play (mid-2026)

- **[Not Diamond](https://www.notdiamond.ai/)** — trains a meta-model over 60+ models predicting which will perform best per query; retrains when new models ship; also does prompt adaptation (rewriting prompts per target model). IBM Ventures backed; SAP integrated it into their Generative AI Hub in 2025. **Powers OpenRouter's auto-router.**
- **[OpenRouter Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router)** (`openrouter/auto`) — Not Diamond under the hood; analyzes the prompt, picks from a curated pool, pins model+provider per session for cache consistency, no routing fee. Relevant to our Phase 3: we can *compare our router against it* as a fun extra baseline.
- **[Martian](https://withmartian.com/)** — routing via predicting model performance without running the model (they frame it as model-internals/interpretability work); published RouterBench; pivoted toward enterprise compliance/governance features.
- **GPT-5's built-in router** ([OpenAI, Aug 2025](https://openai.com/index/introducing-gpt-5/)) — the biggest real-world deployment of exactly this idea: a real-time router choosing between fast and thinking variants, trained on live signals (model-switch events, preference rates, measured correctness). The launch-day "autoswitcher" outage that made GPT-5 "seem way dumber" ([Fortune](https://fortune.com/2025/08/12/openai-gpt-5-model-router-backlash-ai-future/)) is a great cautionary tale for the README: a router is a single point of quality failure.
- **[semantic-router](https://github.com/aurelio-labs/semantic-router)** (Aurelio Labs, MIT) — embedding-similarity routing to *named intents* (kNN over example utterances per route). It routes **by topic, not by difficulty** — adjacent tool, different problem. Worth citing to sharpen what we are NOT building.

---

## 2. The design space: four formulations

### A. Binary difficulty classifier
Label each query "easy" (cheap model suffices) or "hard", train a classifier. Simple, but labels are artifacts of *which* pair of models you chose, and a hard 0/1 output gives you a single operating point — no cost/quality dial.

### B. Win-rate prediction (RouteLLM) ← **recommended**
Learn `P_θ(win_strong | q)` from preference data; route to the weak model when predicted win probability is below threshold **α**. The formal setup from the paper:

- Preference data `D = {(q, l)}` where `l ∈ {strong wins, tie, weak wins}`, maximize likelihood of the win-prediction model.
- **The threshold α is the product feature**: sweeping it traces the entire cost/quality curve. One trained model, every operating point. This is also what makes calibration a first-class citizen — the score has to *mean* something for the threshold to be tunable.

RouteLLM's four architectures, in ascending capability/cost:
1. **Similarity-weighted ranking** — no training; at inference, weight Arena battles by embedding similarity to the query and fit Bradley-Terry coefficients. (Our "smarter kNN" baseline.)
2. **Matrix factorization** — bilinear model of (model embedding × query embedding) → win probability. Best APGR-per-dollar in the paper.
3. **BERT classifier** — fine-tune BERT-base on preferences, logistic head on [CLS]. ~10ms inference, no LLM call needed to route.
4. **Causal LLM classifier** — fine-tuned Llama-3-8B outputting the label as a token. Most capable, heaviest.

### C. Multi-way / per-model quality prediction (Not Diamond, Prompt-to-Leaderboard)
Predict expected quality for each of K models, pick argmax under a cost constraint. More general and more commercial-shaped, but needs per-model training signal for every candidate and a K-way eval story. The binary version is the right first project; this is the natural v2.

### D. Cascades (FrugalGPT, AutoMix)
No upfront prediction: call cheap, **verify the response**, escalate if bad. Pros: the decision sees the actual answer (more information than the query alone). Cons: worst-case latency = cheap + expensive calls; needs a reliable verifier (its own hard ML problem); user-visible double-billing on escalated queries. The 2026 survey frames routing vs cascading as *pre-inference* vs *post-inference* decisions — cleanest way to remember the split.

**Why B over the others for this project:** it produces a trained artifact (deeptte-formula ✓), the threshold sweep produces exactly the cost/quality tradeoff curves and calibration plots we promised (✓), training data is public (✓), the strongest open academic baseline exists to compare against (✓), and it trains in minutes-to-hours on an M-series Mac (✓). Cascades don't yield a "trained router" artifact so much as a verifier + policy; multi-way needs data we'd have to buy.

---

## 3. Features & architectures (what the router actually sees)

- **Off-the-shelf embeddings + small head** (logistic regression / MLP / kNN): semantic-router and RouteLLM's SW-ranking live here. Cheapest, surprisingly strong, runs anywhere.
- **Fine-tuned small encoder** (BERT-class, ~100–400M params): RouteLLM's sweet spot — <10ms routing decisions. In 2026 the obvious pick is ModernBERT-base or similar.
- **Small generative model** (1–8B): Arch-Router, RouteLLM's Llama router. Only worth it if we want the router to explain itself.
- Signals beyond the raw query used by commercial systems: task type, context length, tool-use requirements, session history. (GPT-5's router also consumes live feedback signals — out of scope for us but worth a README mention.)

## 4. Training data (all public)

- **[lmarena-ai/arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k)** — 55k real user battles across 70+ models, A/B/tie labels. There's also a [140k version](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k). RouteLLM's recipe: cluster models into ~10 Elo tiers (top tiers = "strong" class, lower tier = "weak" class) to convert model-pair battles into strong-vs-weak labels; they used ~80k battles pruned to ~65k.
- **Golden-label augmentation**: datasets with verifiable answers (MMLU/GSM8K-style) where "win" = strong got it right and weak didn't. RouteLLM added ~1.5k MMLU validation questions this way and got large APGR gains on OOD benchmarks.
- **LLM-judge augmentation**: RouteLLM also used ~120k Nectar prompts with GPT-4 judgments (~$700 of API spend). We can do a small-scale version with a budget cap, printed costs.
- **[RouterBench dataset](https://arxiv.org/abs/2403.12031)** — 405k precomputed model outputs with quality + cost. Lets us evaluate routing policies **offline for $0** before any live API traffic.

## 5. Evaluation (the honest-numbers section)

From RouteLLM:
- **PGR** (performance gap recovered): `PGR = (r(router) − r(weak)) / (r(strong) − r(weak))` — 0 means you route like always-weak, 1 means you matched always-strong.
- **CPT(x%)**: minimum % of calls sent to the strong model to reach x% PGR — "cost to hit 80% of the quality gap."
- **APGR**: area under the PGR-vs-cost curve (average PGR over the cost spectrum, discretized).

From RouterBench:
- **Cost-quality plane + non-decreasing convex hull**, and **AIQ** (area under the routing curve). Any single point is misleading; the curve is the result.

Plus, because our pitch is calibration:
- **Reliability diagrams and ECE** on the win-rate predictions, and
- **Random-router baseline**: interpolating between always-cheap and always-expensive by coin-flip is the null hypothesis every learned router must beat *above* the line, not just on it.

Baselines for our eval table: always-cheap, always-expensive, random mixture (the diagonal), keyword/length heuristics (my MAIN/MINI-style rules), embedding+LR, then the trained encoder.

## 6. Proposed build (pending sign-off)

1. **Data**: arena-human-preference-55k → tier-based strong/weak relabeling (RouteLLM recipe) + a small golden-label set.
2. **Router v0**: embedding (local, e.g. a small sentence-transformer via MLX/CPU) + logistic regression. Hours of work, establishes the pipeline end to end.
3. **Router v1**: fine-tuned small encoder (ModernBERT-class) on M-series (MPS). Target <10ms routing latency.
4. **Offline eval**: PGR/APGR/CPT curves + calibration + all baselines, on held-out Arena data and at least one OOD benchmark (RouterBench slice or MT-Bench).
5. **Phase 3**: OpenRouter dispatch (cheap = e.g. a small frontier-lab model, expensive = a frontier model — exact pair chosen by current pricing), end-to-end quality/cost/latency vs always-frontier, live demo on Railway showing predicted win-rate, chosen model, running savings.

## Addendum: Weave Router (workweave/router) breakdown

*Added after cloning and reading the source ([github.com/workweave/router](https://github.com/workweave/router), Elastic License v2, Go).*

Weave is a **drop-in proxy** (Anthropic/OpenAI/Gemini wire formats) whose default routing strategy implements the **Avengers-Pro** paper ([arXiv 2508.12631](https://arxiv.org/abs/2508.12631)) — a fifth formulation for our design space:

**E. Cluster-based performance lookup (Avengers-Pro / Weave)**
1. Embed the prompt tail (last 1024 chars) with a small int8 ONNX embedding model (Jina-v2-base-code 768d, newer bundles Qwen3-Embedding-0.6B 1024d) — this is the only "model inference" in the routing path.
2. Find the top-P (P=4) nearest of **16 k-means cluster centroids**.
3. For each candidate model, blend per-cluster scores: `score = α·quality_norm + w_speed·(1−speed_norm) + (1−α−w_speed)·(1−cost_norm)`, summed over the top clusters. Quality comes from a frozen per-cluster × per-model `quality_means.json` table derived from **benchmark evals** (AIDER Polyglot, BFCL, GPQA-Diamond, IFBench, SWE-bench probes — not preference data).
4. Argmax (+ small additive bonuses for subscription-covered and user-preferred models); session pinned to first decision for cache coherence.

So the "pre-trained ML model" is really: an off-the-shelf embedder + k-means centroids + a benchmark-derived lookup table. **There is no per-query learned difficulty/win-rate model.** The learned-router work (RL/DPO policy, HMM, Thompson-sampling bandit) exists as opt-in sidecars, not the default.

Engineering ideas worth stealing:
- **Dial calibration**: their quality-vs-price dial in [0,1] is mapped through precomputed α breakpoints where the routed mix actually changes — no dead zones on the slider. Directly applicable to our threshold α demo UI.
- **Versioned frozen artifacts** (`centroids.bin` + rankings per version, promoted by a one-line pointer change) with multi-version A/B evals.
- **Propensity + candidate-score logging** on every decision — off-policy evaluation substrate.
- Truncate-to-tail + strict embed timeout for predictable p99 routing latency.

Why we don't piggyback for Phase 2:
- **The trainable part isn't public.** `train_cluster_router.py` is referenced but not in the repo; only frozen artifacts are committed. Their own `metadata.yaml` states the training corpus is "no longer reproducible" (source datasets drifted/return 0 rows). The thing this project exists to demonstrate — training — is exactly the part we can't reuse.
- **License**: ELv2 is source-available, not open source; fine to study and self-host, but a portfolio built on it inherits restrictions.
- **Formulation mismatch**: cluster lookup gives no per-query difficulty score, hence no calibration story — our core pitch.

How we *will* use it:
- Implement a **mini Avengers-Pro cluster scorer as a baseline** in our eval table (k-means over embeddings + per-cluster win-rate table — ~50 lines with sklearn). RouteLLM's similarity-weighted router and this are cousins (local non-parametric vs. cluster-quantized lookup); beating both with a trained classifier is a meaningful result.
- Optionally compare against Weave/OpenRouter auto-routing end-to-end in Phase 3.
- Copy the dial-calibration and versioned-artifact patterns for our demo.

## Reading list (in order)

1. **RouteLLM** — [arxiv.org/abs/2406.18665](https://arxiv.org/abs/2406.18665) — the formulation we're adopting; read fully.
2. **RouterBench** — [arxiv.org/abs/2403.12031](https://arxiv.org/abs/2403.12031) — the eval mindset (curves, hulls, AIQ); skim methods, internalize §metrics.
3. **FrugalGPT** — [arxiv.org/abs/2305.05176](https://arxiv.org/abs/2305.05176) — the cascade alternative; read to understand what we're *not* building and why.
4. (Bonus) **2026 survey** — [arxiv.org/abs/2603.04445](https://arxiv.org/abs/2603.04445) — the field map if you want to go deeper.
