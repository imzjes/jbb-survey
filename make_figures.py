"""Produce figures and LaTeX tables for the report from the phase summaries.

Reads `results/phaseN_summary.json` and writes:
  results/fig_phase1_baseline_asr.pdf
  results/fig_phase2_defense_asr.pdf
  results/fig_phase3_refusal.pdf
  results/tables.tex     # Drop-in \\input{} for the report
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

# Colab sets MPLBACKEND to a Jupyter-only backend that fails when this script
# runs outside an IPython kernel. Force a non-interactive backend before
# pyplot is imported so headless PDF output always works.
os.environ.setdefault("MPLBACKEND", "Agg")
if os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils import RESULTS_DIR


def load_summary(name: str) -> list[dict]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    return json.loads(path.read_text())


def fig_phase1(rows: list[dict]) -> None:
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    attacks = sorted({r["attack"] for r in rows})
    width = 0.35
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, attack in enumerate(attacks):
        vals = [next((r["asr"] for r in rows
                      if r["model"] == m and r["attack"] == attack), 0.0)
                for m in models]
        ax.bar(x + i * width, vals, width, label=attack)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Phase 1: Baseline ASR (undefended)")
    ax.set_ylim(0, 1)
    ax.legend(title="Attack")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_phase1_baseline_asr.pdf")
    plt.close(fig)


def fig_phase2(rows_p1: list[dict], rows_p2: list[dict]) -> None:
    if not rows_p2:
        return
    # Group by (model, attack) -> {defense: asr}, plus baseline from Phase 1.
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows_p1:
        grouped[(r["model"], r["attack"])]["None"] = r["asr"]
    for r in rows_p2:
        grouped[(r["model"], r["attack"])][r["defense"]] = r["asr"]

    defense_order = ["None", "SmoothLLM", "PerplexityFilter", "EraseAndCheck"]
    keys = sorted(grouped.keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.18
    x = np.arange(len(keys))
    for i, defense in enumerate(defense_order):
        vals = [grouped[k].get(defense, np.nan) for k in keys]
        ax.bar(x + i * width, vals, width, label=defense)

    ax.set_xticks(x + width * (len(defense_order) - 1) / 2)
    ax.set_xticklabels([f"{m}\n{a}" for (m, a) in keys], fontsize=8)
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Phase 2: Defended ASR vs. Undefended Baseline")
    ax.set_ylim(0, 1)
    ax.legend(title="Defense", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_phase2_defense_asr.pdf")
    plt.close(fig)


def fig_phase3(rows: list[dict]) -> None:
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    defenses = sorted({(r["defense"] or "None") for r in rows},
                      key=lambda d: (d != "None", d))
    width = 0.18
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, defense in enumerate(defenses):
        vals = []
        for m in models:
            match = next(
                (r for r in rows
                 if r["model"] == m and (r["defense"] or "None") == defense),
                None,
            )
            vals.append(match["refusal_rate"] if match else np.nan)
        ax.bar(x + i * width, vals, width, label=defense)

    ax.set_xticks(x + width * (len(defenses) - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Refusal Rate (benign prompts)")
    ax.set_title("Phase 3: Over-refusal on Benign Behaviors")
    ax.set_ylim(0, 1)
    ax.legend(title="Defense", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_phase3_refusal.pdf")
    plt.close(fig)


def write_tables(p1: list[dict], p2: list[dict], p3: list[dict]) -> None:
    lines: list[str] = []

    if p1:
        lines.append("% Phase 1 baseline ASR")
        lines.append("\\begin{tabular}{llrr}")
        lines.append("\\toprule")
        lines.append("Model & Attack & N & ASR \\\\")
        lines.append("\\midrule")
        for r in sorted(p1, key=lambda r: (r["model"], r["attack"])):
            lines.append(f"{r['model']} & {r['attack']} & "
                         f"{r['n_prompts']} & {r['asr']:.3f} \\\\")
        lines.append("\\bottomrule\n\\end{tabular}\n")

    if p2:
        lines.append("% Phase 2 defended ASR")
        lines.append("\\begin{tabular}{lllr}")
        lines.append("\\toprule")
        lines.append("Model & Attack & Defense & ASR \\\\")
        lines.append("\\midrule")
        for r in sorted(p2, key=lambda r: (r["model"], r["attack"], r["defense"])):
            lines.append(f"{r['model']} & {r['attack']} & "
                         f"{r['defense']} & {r['asr']:.3f} \\\\")
        lines.append("\\bottomrule\n\\end{tabular}\n")

    if p3:
        lines.append("% Phase 3 benign refusal")
        lines.append("\\begin{tabular}{llr}")
        lines.append("\\toprule")
        lines.append("Model & Defense & Refusal Rate \\\\")
        lines.append("\\midrule")
        for r in sorted(p3, key=lambda r: (r["model"], r["defense"] or "")):
            d = r["defense"] or "None"
            lines.append(f"{r['model']} & {d} & {r['refusal_rate']:.3f} \\\\")
        lines.append("\\bottomrule\n\\end{tabular}\n")

    (RESULTS_DIR / "tables.tex").write_text("\n".join(lines))


def main() -> None:
    p1 = load_summary("phase1_summary.json")
    p2 = load_summary("phase2_summary.json")
    p3 = load_summary("phase3_summary.json")
    fig_phase1(p1)
    fig_phase2(p1, p2)
    fig_phase3(p3)
    write_tables(p1, p2, p3)
    print("Figures + tables written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
