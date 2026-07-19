# Phase 2 v0 Implementation Plan — Learned Win-Rate Router

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline execution chosen — the human is learning from each step).

**Goal:** Train a binary win-rate router `P(strong wins | query)` on Chatbot Arena preference data, evaluated with cost/quality tradeoff curves against honest baselines.

**Architecture:** RouteLLM formulation. Data pipeline converts Arena battles into strong-vs-weak labels via Elo tiering. v0 router = frozen sentence-transformer embeddings + logistic regression. Offline eval on held-out battles: PGR/APGR/CPT + calibration + baseline table.

**Tech Stack:** Python 3.12, uv, HuggingFace `datasets`, `sentence-transformers` (all-MiniLM-L6-v2), scikit-learn, pandas/pyarrow, matplotlib.

---

## File structure

```
src/router/
  elo.py          # Bradley-Terry/Elo fit on battles; tier assignment
  labels.py       # battle -> (prompt, strong_wins) label mapping (pure functions)
  embed.py        # embedding cache: prompts -> npy
  evaluate.py     # PGR/APGR/CPT, calibration, baselines
scripts/
  prepare_data.py # download 55k -> elo -> tiers -> labeled parquet splits
  train_v0.py     # embeddings + logistic regression -> artifacts/v0/
  eval_v0.py      # eval table + curves -> artifacts/v0/
tests/
  test_labels.py  # label mapping unit tests
  test_elo.py     # elo/tier sanity tests
```

## Tasks

### Task 1: Scaffold
- [x] `git init`, `uv init --python 3.12`, add deps, `.gitignore` (data/, *.npy, .venv), MIT license
- [x] Commit

### Task 2: Data pipeline (TDD on the pure functions)
- [x] Failing tests for `labels.py`: cross-class battle -> label 1/0; same-class battle -> None; tie -> None
- [x] Implement `elo.py` (logistic-regression Bradley-Terry), `labels.py`
- [x] `scripts/prepare_data.py`: download `lmarena-ai/arena-human-preference-55k`, fit Elo, 10 tiers (1-D k-means, RouteLLM-style), strong = tiers 1-2, weak = tier 3; emit `data/battles_labeled.parquet` + train/val/test splits (stratified, by prompt) + `data/tiers.json`
- [x] Run it; record row counts (printed, go in README)
- [x] Commit

### Task 3: v0 router
- [x] `embed.py`: batch-encode prompts with all-MiniLM-L6-v2 (MPS if available), cache to `data/emb_*.npy`
- [x] `train_v0.py`: logistic regression (sklearn, class-weighted) on train, report val AUC/acc; save `artifacts/v0/model.joblib`
- [x] Commit

### Task 4: Honest eval
- [x] `evaluate.py`: threshold sweep -> (cost = frac routed strong, quality = frac routed side won battle); PGR/CPT/APGR; ECE + reliability diagram; baselines: always-cheap, always-expensive, random mixture, length heuristic
- [x] `eval_v0.py` writes `artifacts/v0/eval.json` + curves PNG
- [x] Commit

### Task 5: Publish
- [x] README: project story, formulation, eval table (ONLY numbers from our runs), limitations
- [x] `gh repo create PeterP22/tiered-model-router --public`, push

## Explicitly deferred (v1+)
- Fine-tuned encoder (ModernBERT-class) router
- Golden-label augmentation (MMLU-style)
- Cluster-scorer (Avengers-Pro-style) baseline
- Phase 3: OpenRouter live dispatch + Railway demo
