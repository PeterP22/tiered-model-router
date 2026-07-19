"""Embed prompts with a frozen sentence-transformer, cached to .npy files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_split(df: pd.DataFrame, cache_dir: Path, split: str) -> np.ndarray:
    """Encode df.prompt_text; cache keyed on split name + content hash."""
    digest = hashlib.sha256("\x00".join(df["prompt_text"]).encode()).hexdigest()[:12]
    cache = cache_dir / f"emb_{split}_{digest}.npy"
    if cache.exists():
        return np.load(cache)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        df["prompt_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(cache, emb)
    return emb
