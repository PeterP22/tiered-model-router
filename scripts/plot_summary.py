"""Bar chart of cpt_50 / cpt_80 per router, read only from artifacts/v1/eval.json.

No API keys or model needed: `uv run python scripts/plot_summary.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_JSON = ROOT / "artifacts" / "v1" / "eval.json"
OUT_PNG = ROOT / "docs" / "routing_summary.png"

POLICIES = [
    ("v1_embed_lr", "v1 router", "#2a78d6"),
    ("length_heuristic", "length heuristic", "#eb6834"),
    ("random", "random", "#1baf7a"),
]
TARGETS = [("cpt_50", "50% of gap"), ("cpt_80", "80% of gap")]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def summary_series(results: dict) -> dict[str, list[float]]:
    """Plotted values: label -> [cpt_50, cpt_80] for each policy."""
    return {label: [results[key][t] for t, _ in TARGETS] for key, label, _ in POLICIES}


def plot(eval_path: Path = EVAL_JSON, out_path: Path = OUT_PNG) -> dict[str, list[float]]:
    results = json.loads(Path(eval_path).read_text())
    series = summary_series(results)

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(TARGETS))
    width = 0.26
    for i, (_, label, color) in enumerate(POLICIES):
        vals = series[label]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color,
                      edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=3,
                     color=INK, fontsize=9)

    ax.set_xticks(x, [name for _, name in TARGETS], color=INK)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylabel("share of queries sent to Opus", color=MUTED)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

    fig.suptitle("Share of queries sent to Opus to recover 50% / 80% of the quality gap",
                 x=0.02, ha="left", fontsize=11, color=INK)
    ax.set_title(f"Haiku 4.5 vs Opus 4.8, held-out test n={results['n_test']}. Lower is better.",
                 loc="left", fontsize=9, color=MUTED)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return series


if __name__ == "__main__":
    print(json.dumps(plot(), indent=2))
