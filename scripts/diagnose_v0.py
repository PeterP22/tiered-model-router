"""Diagnose the v0 chance-level AUC: pipeline bug vs. genuine label noise.

Evidence gathered in one run:
  A. Train AUC of the shipped model (memorization capacity on real labels).
  B. Probe task: predict "prompt contains code fence / length > median" from
     the same embeddings (validates embeddings+alignment end to end).
  C. Shuffled-label control (what memorization looks like with zero signal).
  D. Clean-pair subset: GPT-4-family strong vs (mixtral / gpt-3.5 / llama-70b)
     weak — 5-fold CV AUC. RouteLLM's arena eval was a single fixed pair;
     heterogeneous pairs may be washing out per-query signal.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from router.embed import embed_split

ROOT = Path(__file__).resolve().parent.parent
DATA, ART = ROOT / "data", ROOT / "artifacts" / "v0"

STRONG_CLEAN = {"gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-0613", "gpt-4-0314"}
WEAK_CLEAN = {"mixtral-8x7b-instruct-v0.1", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-0613", "llama-2-70b-chat"}


def lr() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")


def main() -> None:
    train = pd.read_parquet(DATA / "split_train.parquet")
    X = embed_split(train, DATA, "train")
    y = train["label"].to_numpy()

    # A. memorization on real labels
    clf = joblib.load(ART / "model.joblib")
    print(f"A. train AUC (real labels): {roc_auc_score(y, clf.predict_proba(X)[:, 1]):.4f}")

    # B. probe: is the feature pipeline capable of learning anything aligned?
    has_code = train["prompt_text"].str.contains("```|def |import ", regex=True).to_numpy()
    long_prompt = (train["prompt_text"].str.len() > train["prompt_text"].str.len().median()).to_numpy()
    for name, target in [("has_code", has_code), ("long_prompt", long_prompt)]:
        auc = cross_val_score(lr(), X, target, cv=5, scoring="roc_auc").mean()
        print(f"B. probe 5-fold AUC ({name}): {auc:.4f}")

    # C. shuffled-label control
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    fit = lr().fit(X, y_shuf)
    print(f"C. train AUC (shuffled labels): {roc_auc_score(y_shuf, fit.predict_proba(X)[:, 1]):.4f}")
    cv_shuf = cross_val_score(lr(), X, y_shuf, cv=5, scoring="roc_auc").mean()
    print(f"C. CV AUC (shuffled labels): {cv_shuf:.4f}")

    # D. clean fixed-pair subset (both directions a/b)
    full = pd.read_parquet(DATA / "battles_labeled.parquet")
    clean = full[
        (full["model_a"].isin(STRONG_CLEAN) & full["model_b"].isin(WEAK_CLEAN))
        | (full["model_b"].isin(STRONG_CLEAN) & full["model_a"].isin(WEAK_CLEAN))
    ].reset_index(drop=True)
    print(f"D. clean-pair battles: {len(clean)}, P(strong wins)={clean['label'].mean():.3f}")
    Xc = embed_split(clean, DATA, "clean_pairs")
    cv = cross_val_score(lr(), Xc, clean["label"], cv=5, scoring="roc_auc")
    print(f"D. clean-pair 5-fold CV AUC: {cv.mean():.4f} +/- {cv.std():.4f}")

    # D2. full-data CV as reference (rules out an unlucky split)
    cv_full = cross_val_score(lr(), X, y, cv=5, scoring="roc_auc")
    print(f"D2. full train 5-fold CV AUC: {cv_full.mean():.4f} +/- {cv_full.std():.4f}")


if __name__ == "__main__":
    main()
