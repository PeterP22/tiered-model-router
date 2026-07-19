"""Train router v0: frozen MiniLM embeddings + logistic regression."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from router.embed import embed_split

ROOT = Path(__file__).resolve().parent.parent
DATA, ART = ROOT / "data", ROOT / "artifacts" / "v0"


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DATA / "split_train.parquet")
    val = pd.read_parquet(DATA / "split_val.parquet")

    X_train = embed_split(train, DATA, "train")
    X_val = embed_split(val, DATA, "val")

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X_train, train["label"])

    p_val = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(val["label"], p_val)
    acc = accuracy_score(val["label"], p_val > 0.5)
    print(f"val AUC: {auc:.4f}")
    print(f"val acc@0.5: {acc:.4f}")
    print(f"val base rate P(strong wins): {val['label'].mean():.4f}")

    joblib.dump(clf, ART / "model.joblib")
    print(f"saved {ART}/model.joblib")


if __name__ == "__main__":
    main()
