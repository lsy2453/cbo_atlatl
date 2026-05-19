"""
Publication-style plots for Atlatl CBO experiments.

Examples:
    python cbo_framework/plot_results.py --results passagg_results.json qwen_results_fixed.json --labels pass-agg qwen
    python cbo_framework/plot_results.py --results qwen_results.json --labels qwen
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
]


def set_publication_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_result(path, label, n_initial=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scores = np.asarray(data.get("all_scores", []), dtype=float)
    if scores.size == 0:
        raise ValueError(f"No all_scores found in {path}")
    best_trace = np.minimum.accumulate(scores)
    best_y = float(data.get("best_y", np.min(scores)))
    inferred_initial = data.get("n_initial", n_initial)
    if inferred_initial is None:
        # Historical runs did not store n_initial. The current default experiment
        # used by this project is 10 initial random interventions.
        inferred_initial = min(10, len(scores))
    inferred_initial = int(min(max(inferred_initial, 1), len(scores)))
    return {
        "path": path,
        "label": label,
        "scores": scores,
        "best_trace": best_trace,
        "best_y": best_y,
        "best_x": data.get("best_x", {}),
        "n_evaluations": int(data.get("n_evaluations", len(scores))),
        "n_initial": inferred_initial,
    }


def short_label(path):
    name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ["_results_fixed", "_results", "_small", "_smoke"]:
        name = name.replace(suffix, "")
    return name


def save_figure(fig, out_dir, stem, timestamp):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{stem}_{timestamp}.png")
    pdf = os.path.join(out_dir, f"{stem}_{timestamp}.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def plot_overview(results, out_dir, timestamp):
    fig = plt.figure(figsize=(10.8, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.28)
    ax_trace = fig.add_subplot(gs[0, :])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    # A. Optimization trace
    for i, result in enumerate(results):
        color = COLORS[i % len(COLORS)]
        evals = np.arange(1, len(result["scores"]) + 1)
        ax_trace.scatter(
            evals, result["scores"], s=18, color=color, alpha=0.28,
            edgecolor="none"
        )
        ax_trace.plot(
            evals, result["best_trace"], color=color,
            label=f"{result['label']} best-so-far"
        )
    ax_trace.set_title("A. Adversarial CBO optimization trace")
    ax_trace.set_xlabel("Evaluation")
    ax_trace.set_ylabel("Blue score (lower is more adversarial)")
    ax_trace.axhline(0, color="#666666", lw=0.8, ls="--", alpha=0.6)
    ax_trace.grid(True, color="#E6E6E6", lw=0.7)
    ax_trace.legend(frameon=False, ncol=min(len(results), 3), loc="best")

    # B. Score distribution
    score_lists = [r["scores"] for r in results]
    parts = ax_dist.violinplot(score_lists, showmeans=False, showmedians=False,
                               showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[i % len(COLORS)])
        body.set_edgecolor("none")
        body.set_alpha(0.35)
    ax_dist.boxplot(
        score_lists, widths=0.18, patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.8},
        medianprops={"color": "#333333", "linewidth": 1.2},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        capprops={"color": "#333333", "linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.35},
    )
    ax_dist.set_title("B. Evaluated score distribution")
    ax_dist.set_ylabel("Blue score")
    ax_dist.set_xticks(np.arange(1, len(results) + 1))
    ax_dist.set_xticklabels([r["label"] for r in results], rotation=20, ha="right")
    ax_dist.axhline(0, color="#666666", lw=0.8, ls="--", alpha=0.6)
    ax_dist.grid(True, axis="y", color="#E6E6E6", lw=0.7)

    # C. Final best comparison
    labels = [r["label"] for r in results]
    best_values = [r["best_y"] for r in results]
    bars = ax_bar.bar(
        np.arange(len(results)), best_values,
        color=[COLORS[i % len(COLORS)] for i in range(len(results))],
        alpha=0.86,
    )
    ax_bar.set_title("C. Worst scenario found")
    ax_bar.set_ylabel("Minimum observed Blue score")
    ax_bar.set_xticks(np.arange(len(results)))
    ax_bar.set_xticklabels(labels, rotation=20, ha="right")
    ax_bar.axhline(0, color="#666666", lw=0.8, ls="--", alpha=0.6)
    ax_bar.grid(True, axis="y", color="#E6E6E6", lw=0.7)
    for bar, value in zip(bars, best_values):
        va = "top" if value < 0 else "bottom"
        offset = -4 if value < 0 else 4
        ax_bar.annotate(
            f"{value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )

    fig.suptitle("Adversarial Scenario Search in ATLATL", y=0.99, fontsize=12)
    return save_figure(fig, out_dir, "cbo_overview", timestamp)


def plot_single_experiment(result, out_dir, timestamp):
    scores = result["scores"]
    best_trace = result["best_trace"]
    n_initial = result["n_initial"]
    evals = np.arange(1, len(scores) + 1)
    best_idx = int(np.argmin(scores))

    fig = plt.figure(figsize=(10.8, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.30)
    ax_trace = fig.add_subplot(gs[0, :])
    ax_sorted = fig.add_subplot(gs[1, 0])
    ax_phase = fig.add_subplot(gs[1, 1])

    initial_mask = evals <= n_initial
    cbo_mask = evals > n_initial
    ax_trace.scatter(
        evals[initial_mask], scores[initial_mask], s=28, color="#8DBBD9",
        edgecolor="white", linewidth=0.4, label="initial random evaluations"
    )
    if np.any(cbo_mask):
        ax_trace.scatter(
            evals[cbo_mask], scores[cbo_mask], s=28, color="#D55E00",
            edgecolor="white", linewidth=0.4, label="CBO-selected evaluations"
        )
    ax_trace.plot(evals, best_trace, color="#0072B2", lw=2.2, label="best-so-far")
    ax_trace.scatter(
        [best_idx + 1], [scores[best_idx]], marker="*", s=180,
        color="#C00000", edgecolor="white", linewidth=0.8, zorder=5,
        label="best adversarial scenario"
    )
    if n_initial < len(scores):
        ax_trace.axvline(n_initial + 0.5, color="#666666", lw=1.0, ls=":",
                         label="CBO starts")
    ax_trace.axhline(0, color="#888888", lw=0.8, ls="--", alpha=0.7)
    ax_trace.set_title("A. Optimization trace with random/CBO phases")
    ax_trace.set_xlabel("Evaluation")
    ax_trace.set_ylabel("Blue score (lower is more adversarial)")
    ax_trace.grid(True, color="#E6E6E6", lw=0.7)
    ax_trace.legend(frameon=False, ncol=2, loc="best")

    sorted_scores = np.sort(scores)
    ranks = np.arange(1, len(scores) + 1)
    ax_sorted.plot(ranks, sorted_scores, color="#333333", lw=1.8)
    ax_sorted.scatter(ranks[:5], sorted_scores[:5], color="#C00000", s=24, zorder=3)
    ax_sorted.axhline(0, color="#888888", lw=0.8, ls="--", alpha=0.7)
    ax_sorted.set_title("B. Ranked adversarial severity")
    ax_sorted.set_xlabel("Rank among evaluated scenarios")
    ax_sorted.set_ylabel("Blue score")
    ax_sorted.grid(True, color="#E6E6E6", lw=0.7)

    initial_best = float(np.min(scores[:n_initial]))
    final_best = float(np.min(scores))
    cbo_best = float(np.min(scores[n_initial:])) if n_initial < len(scores) else np.nan
    bars = [initial_best, final_best]
    labels = ["initial best", "final best"]
    colors = ["#8DBBD9", "#0072B2"]
    if np.isfinite(cbo_best):
        bars.insert(1, cbo_best)
        labels.insert(1, "CBO best")
        colors.insert(1, "#D55E00")
    ax_phase.bar(np.arange(len(bars)), bars, color=colors, alpha=0.9)
    ax_phase.axhline(0, color="#888888", lw=0.8, ls="--", alpha=0.7)
    ax_phase.set_title("C. Improvement over random search")
    ax_phase.set_ylabel("Minimum observed Blue score")
    ax_phase.set_xticks(np.arange(len(bars)))
    ax_phase.set_xticklabels(labels, rotation=15, ha="right")
    ax_phase.grid(True, axis="y", color="#E6E6E6", lw=0.7)
    for idx, val in enumerate(bars):
        va = "top" if val < 0 else "bottom"
        offset = -4 if val < 0 else 4
        ax_phase.annotate(
            f"{val:.1f}", xy=(idx, val), xytext=(0, offset),
            textcoords="offset points", ha="center", va=va, fontsize=8
        )

    fig.suptitle(f"Adversarial CBO Diagnostic: {result['label']}",
                 y=0.99, fontsize=12)
    return save_figure(fig, out_dir, "single_experiment_diagnostic", timestamp)


def plot_best_scenarios(results, out_dir, timestamp):
    numeric_vars = [
        "n_blue", "n_red", "max_phases", "p_urban", "p_rough", "p_marsh"
    ]
    ranges = {
        "n_blue": (1, 4),
        "n_red": (1, 4),
        "max_phases": (6, 20),
        "p_urban": (0.0, 0.5),
        "p_rough": (0.0, 0.5),
        "p_marsh": (0.0, 0.3),
    }

    values = []
    raw_values = []
    for result in results:
        row = []
        raw_row = []
        for var in numeric_vars:
            val = float(result["best_x"].get(var, np.nan))
            lo, hi = ranges[var]
            row.append((val - lo) / (hi - lo))
            raw_row.append(val)
        values.append(row)
        raw_values.append(raw_row)

    values = np.asarray(values, dtype=float)
    raw_values = np.asarray(raw_values, dtype=float)

    fig, ax = plt.subplots(figsize=(9.6, max(2.6, 0.55 * len(results) + 1.8)))
    im = ax.imshow(values, aspect="auto", cmap="viridis_r", vmin=0, vmax=1)
    ax.set_title("Best adversarial scenario parameters (normalized)")
    ax.set_xticks(np.arange(len(numeric_vars)))
    ax.set_xticklabels(numeric_vars, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(results)))
    ax.set_yticklabels([r["label"] for r in results])

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = raw_values[i, j]
            text = f"{val:.2f}" if numeric_vars[j].startswith("p_") else f"{val:.0f}"
            ax.text(j, i, text, ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normalized value within search range")

    categorical_lines = []
    for result in results:
        x = result["best_x"]
        categorical_lines.append(
            f"{result['label']}: blue={x.get('blue_unit_type')}@{x.get('blue_side')}, "
            f"red={x.get('red_unit_type')} / {x.get('red_ai')}, "
            f"best_y={result['best_y']:.1f}"
        )
    fig.text(
        0.02, -0.02, "\n".join(categorical_lines),
        ha="left", va="top", fontsize=8
    )
    return save_figure(fig, out_dir, "best_scenario_parameters", timestamp)


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-style plots for CBO result JSON files."
    )
    parser.add_argument("--results", nargs="+", required=True,
                        help="result JSON files from cbo_framework/run.py")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="display labels, one per result file")
    parser.add_argument("--out_dir", default="figures",
                        help="directory for timestamped PNG/PDF figures")
    parser.add_argument("--timestamp", default=None,
                        help="override timestamp, default YYYYMMDD_HHMMSS")
    parser.add_argument("--n_initial", type=int, default=None,
                        help="initial random evaluations for legacy result files")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.results):
        raise ValueError("--labels must have the same length as --results")

    labels = args.labels or [short_label(path) for path in args.results]
    results = [
        load_result(path, label, n_initial=args.n_initial)
        for path, label in zip(args.results, labels)
    ]
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    set_publication_style()
    saved = []
    if len(results) == 1:
        saved.extend(plot_single_experiment(results[0], args.out_dir, timestamp))
    else:
        saved.extend(plot_overview(results, args.out_dir, timestamp))
    saved.extend(plot_best_scenarios(results, args.out_dir, timestamp))

    print("Saved figures:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
