"""
Publication-style plots for method x AI benchmark JSON files.

Example:
    python cbo_framework/plot_benchmark.py --benchmark benchmark_results/benchmark_*.json
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "random_search",
    "standard_bo",
    "neighborhood",
    "sobol_guided",
    "causal_bo",
]
METHOD_LABELS = {
    "random_search": "Random",
    "standard_bo": "Standard BO",
    "neighborhood": "Neighborhood",
    "sobol_guided": "Sobol-guided",
    "causal_bo": "Causal BO",
}
COLORS = {
    "random_search": "#999999",
    "standard_bo": "#0072B2",
    "neighborhood": "#D55E00",
    "sobol_guided": "#009E73",
    "causal_bo": "#CC79A7",
}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_runs(paths):
    runs = []
    for pattern in paths:
        for path in glob.glob(pattern):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for run in payload.get("runs", []):
                if run.get("status") == "failed":
                    continue
                if "best_y" in run:
                    run = dict(run)
                    run["_source"] = path
                    runs.append(run)
    if not runs:
        raise ValueError("No successful benchmark runs found.")
    return runs


def save(fig, out_dir, stem, timestamp):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{stem}_{timestamp}.png")
    pdf = os.path.join(out_dir, f"{stem}_{timestamp}.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def summarize_best(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["blue_ai"], run["method"])].append(float(run["best_y"]))
    summary = {}
    for key, vals in grouped.items():
        vals = np.asarray(vals, dtype=float)
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        }
    return summary


def plot_bars_and_heatmap(runs, out_dir, timestamp):
    summary = summarize_best(runs)
    ais = sorted({run["blue_ai"] for run in runs})
    methods = [m for m in METHOD_ORDER if any(run["method"] == m for run in runs)]

    fig = plt.figure(figsize=(11.4, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.34)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    width = 0.78 / max(len(methods), 1)
    x = np.arange(len(ais))
    for j, method in enumerate(methods):
        means = [summary.get((ai, method), {}).get("mean", np.nan) for ai in ais]
        stds = [summary.get((ai, method), {}).get("std", 0.0) for ai in ais]
        offset = (j - (len(methods) - 1) / 2) * width
        ax_bar.bar(
            x + offset, means, width=width,
            yerr=stds, capsize=2.5,
            color=COLORS.get(method, "#777777"),
            label=METHOD_LABELS.get(method, method),
            alpha=0.9,
        )
    ax_bar.axhline(0, color="#777777", ls="--", lw=0.8)
    ax_bar.set_title("A. Worst score by AI and search method")
    ax_bar.set_ylabel("Minimum Blue score (lower is more adversarial)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ais, rotation=20, ha="right")
    ax_bar.grid(True, axis="y", color="#E6E6E6", lw=0.7)
    ax_bar.legend(frameon=False, ncol=2)

    mat = np.full((len(ais), len(methods)), np.nan)
    for i, ai in enumerate(ais):
        for j, method in enumerate(methods):
            if (ai, method) in summary:
                mat[i, j] = summary[(ai, method)]["mean"]
    im = ax_heat.imshow(mat, aspect="auto", cmap="viridis_r")
    ax_heat.set_title("B. Adversarial severity heatmap")
    ax_heat.set_xticks(np.arange(len(methods)))
    ax_heat.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                            rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(len(ais)))
    ax_heat.set_yticklabels(ais)
    for i in range(len(ais)):
        for j in range(len(methods)):
            if np.isfinite(mat[i, j]):
                ax_heat.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                             color="white", fontsize=8, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Mean minimum Blue score")

    fig.suptitle("ATLATL Adversarial Search Benchmark", fontsize=12, y=0.98)
    return save(fig, out_dir, "benchmark_method_ai", timestamp)


def plot_convergence(runs, out_dir, timestamp):
    ais = sorted({run["blue_ai"] for run in runs})
    methods = [m for m in METHOD_ORDER if any(run["method"] == m for run in runs)]
    fig, axes = plt.subplots(
        len(ais), 1, figsize=(9.6, max(3.0, 2.4 * len(ais))), sharex=True
    )
    if len(ais) == 1:
        axes = [axes]

    for ax, ai in zip(axes, ais):
        for method in methods:
            traces = []
            for run in runs:
                if run["blue_ai"] != ai or run["method"] != method:
                    continue
                scores = np.asarray(run.get("all_scores", []), dtype=float)
                if scores.size:
                    traces.append(np.minimum.accumulate(scores))
            if not traces:
                continue
            min_len = min(len(t) for t in traces)
            arr = np.vstack([t[:min_len] for t in traces])
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(min_len)
            xs = np.arange(1, min_len + 1)
            color = COLORS.get(method, "#777777")
            ax.plot(xs, mean, color=color, label=METHOD_LABELS.get(method, method))
            if arr.shape[0] > 1:
                ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.14)
        ax.axhline(0, color="#777777", ls="--", lw=0.8)
        ax.set_title(f"Blue AI: {ai}")
        ax.set_ylabel("Best-so-far score")
        ax.grid(True, color="#E6E6E6", lw=0.7)
        ax.legend(frameon=False, ncol=3)

    axes[-1].set_xlabel("Evaluation")
    fig.suptitle("CBO Benchmark Convergence", fontsize=12, y=0.995)
    return save(fig, out_dir, "benchmark_convergence", timestamp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", nargs="+", required=True,
                        help="benchmark JSON file(s), glob patterns allowed")
    parser.add_argument("--out_dir", default="figures")
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    set_style()
    runs = load_runs(args.benchmark)
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    saved.extend(plot_bars_and_heatmap(runs, args.out_dir, timestamp))
    saved.extend(plot_convergence(runs, args.out_dir, timestamp))
    print("Saved figures:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
