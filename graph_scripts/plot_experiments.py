#!/usr/bin/env python3
"""Generate verification-category bar charts for experiment outputs.

Produces five PNG files in `output/` for the comparisons requested by the user.

Usage:
  python3 scripts/plot_experiments.py

Dependencies: matplotlib, numpy
Install with: python3 -m pip install matplotlib numpy
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"

# Map of experiment metadata copied from run_experiments.sh arrays
EXPERIMENT_META = {
    1: {"target": "llama4", "verifier": "gemini2.5-flash", "prompt_format": "query_before_context", "use_system": False, "num_queries": 500},
    2: {"target": "llama4", "verifier": "gemini2.5-flash", "prompt_format": "query_before_context", "use_system": True, "num_queries": 500},
    3: {"target": "llama4", "verifier": "gemini2.5-flash", "prompt_format": "context_before_query", "use_system": False, "num_queries": 500},
    4: {"target": "gemini2.5-flash", "verifier": "gemini2.5-flash", "prompt_format": "query_before_context", "use_system": False, "num_queries": 500},
    5: {"target": "llama3.3", "verifier": "gemini2.5-flash", "prompt_format": "query_before_context", "use_system": False, "num_queries": 500},
    6: {"target": "llama4", "verifier": "llama4", "prompt_format": "query_before_context", "use_system": False, "num_queries": 500},
    7: {"target": "llama4", "verifier": "llama3.3", "prompt_format": "query_before_context", "use_system": False, "num_queries": 500},
}

KEEP = ["benign", "adversarial", "refusal"]


def read_counts(exp_num: int) -> Counter:
    path = OUTPUT_DIR / f"experiment_{exp_num}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    c = Counter()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cat = obj.get("verification_category") or obj.get("verification") or obj.get("category")
            if not cat:
                # try inside verification object
                v = obj.get("verification" )
                if isinstance(v, dict):
                    cat = v.get("category")
            if not cat:
                # fallback: skip
                continue
            cat = str(cat).strip().lower()
            if cat not in KEEP:
                # normalize common forms
                if "adv" in cat:
                    cat = "adversarial"
                elif "refus" in cat:
                    cat = "refusal"
                elif "ben" in cat:
                    cat = "benign"
                else:
                    # ignore other categories
                    continue
            c[cat] += 1
    # Ensure all keys present
    for k in KEEP:
        c.setdefault(k, 0)
    return c


def plot_comparison(exp_nums: list[int], outname: str, title: str):
    # Collect counts
    counts = [read_counts(e) for e in exp_nums]

    categories = KEEP
    x = np.arange(len(categories))
    n = len(exp_nums)
    width = 0.8 / max(1, n)

    fig, ax = plt.subplots(figsize=(8, 5))

    offsets = (np.arange(n) - (n - 1) / 2) * width

    colors = plt.cm.tab10.colors

    for i, (exp_num, offset) in enumerate(zip(exp_nums, offsets)):
        vals = [counts[i][cat] for cat in categories]
        bar_positions = x + offset
        label = f"Exp {exp_num} — target={EXPERIMENT_META.get(exp_num,{}).get('target','?')}, verifier={EXPERIMENT_META.get(exp_num,{}).get('verifier','?')}"
        bars = ax.bar(bar_positions, vals, width=width, label=label, color=colors[i % len(colors)])

        # Annotate bars with counts
        for rect in bars:
            height = rect.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    outpath = OUTPUT_DIR / outname
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Wrote {outpath}")
    plt.close(fig)


def main():
    # 1. Normal setup with llama4 (Experiment 1)
    plot_comparison([1], "plot_1_single_llama4.png", "Experiment 1 — Normal setup (target=llama4, verifier=gemini2.5-flash)")

    # 2. System prompt's importance (Experiment 1 vs 2)
    plot_comparison([1, 2], "plot_2_system_prompt.png", "Experiment 1 vs 2 — System prompt importance (use_system: Exp1=False, Exp2=True)")

    # 3. Ordering of user query and context (Experiment 1 vs 3)
    plot_comparison([1, 3], "plot_3_ordering.png", "Experiment 1 vs 3 — Ordering: query/context")

    # 4. Comparison among target models (Exp 1 vs 4 vs 5)
    plot_comparison([1, 4, 5], "plot_4_target_models.png", "Exp 1 vs 4 vs 5 — Target models comparison (verifier=gemini2.5-flash)")

    # 5. Changing verifier model (Exp 1 vs 6 vs 7)
    plot_comparison([1, 6, 7], "plot_5_verifier_models.png", "Exp 1 vs 6 vs 7 — Verifier models (target=llama4)")


if __name__ == "__main__":
    main()
