"""
Supplementary causal-analysis figures for the ATLATL benchmark.

Figures produced:
- Figure 3: ATE forest plot for the LLM CBO surrogate.
- Figure 4: radar chart comparing vulnerability profiles across Blue AIs.

Example:
    python cbo_framework/plot_causal_supplements.py ^
      --benchmark benchmark_results/benchmark_20260519_100605.json ^
      --out_dir figures
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from acquisition import generate_reference_samples
from atlatl_evaluator import vars_to_vector
from benchmark_methods import default_causal_lengthscales
from causal_gp import CausalGP, RBFKernel


AI_LABELS = {
    "agg": "Aggressive",
    "pass-agg": "Pass-Agg",
    "llm-qwen": "Qwen-LLM",
}

AI_COLORS = {
    "agg": "#E69F00",
    "pass-agg": "#0072B2",
    "llm-qwen": "#CC79A7",
}

VAR_LABELS = {
    "n_blue": "Blue count",
    "blue_side": "Blue side",
    "blue_unit_type": "Blue type",
    "n_red": "Red count",
    "red_ai": "Red AI",
    "max_phases": "Horizon",
    "p_urban": "Urban prob.",
    "p_rough": "Rough prob.",
    "p_marsh": "Marsh prob.",
    "red_unit_type": "Red type",
}


def load_runs(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", payload.get("results", []))
    return [r for r in runs if "best_y" in r and "all_scores" in r and "all_x" in r], payload


def collect_training_data(runs, blue_ai, method=None):
    xs = []
    ys = []
    for run in runs:
        if run.get("blue_ai") != blue_ai:
            continue
        if method is not None and run.get("method") != method:
            continue
        for x, y in zip(run.get("all_x", []), run.get("all_scores", [])):
            xs.append(vars_to_vector(x))
            ys.append(float(y))
    if not xs:
        raise ValueError(f"No observations found for blue_ai={blue_ai}, method={method}.")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def fit_gp(X, y, causal=True):
    lengthscale = default_causal_lengthscales() if causal else np.ones(len(config.VAR_NAMES))
    gp = CausalGP(RBFKernel(lengthscale=lengthscale, noise_var=1e-3))
    gp.fit(X, y)
    return gp


def candidate_values(name):
    spec = config.SCENARIO_VARS[name]
    if spec["type"] == "categorical":
        return list(range(len(spec["categories"]))), list(spec["categories"])
    if spec["type"] == "integer":
        return list(range(spec["low"], spec["high"] + 1)), [
            str(v) for v in range(spec["low"], spec["high"] + 1)
        ]
    return [spec["low"], spec["high"]], [
        f"{spec['low']:.2f}",
        f"{spec['high']:.2f}",
    ]


def intervention_predictions(gp, var_name, values, reference_samples):
    idx = config.VAR_INDICES[var_name]
    means = []
    sample_mus = []
    for value in values:
        X_do = np.array(reference_samples, dtype=float, copy=True)
        X_do[:, idx] = value
        mu, _ = gp.predict(X_do)
        means.append(float(np.mean(mu)))
        sample_mus.append(mu)
    return np.asarray(means), sample_mus


def variable_effect_summary(gp, var_name, reference_samples):
    values, labels = candidate_values(var_name)
    means, sample_mus = intervention_predictions(gp, var_name, values, reference_samples)
    adverse_i = int(np.argmin(means))
    benign_i = int(np.argmax(means))
    diffs = sample_mus[adverse_i] - sample_mus[benign_i]
    return {
        "variable": var_name,
        "label": VAR_LABELS.get(var_name, var_name),
        "from": labels[benign_i],
        "to": labels[adverse_i],
        "effect": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "range": float(np.max(means) - np.min(means)),
        "means": means.tolist(),
        "levels": labels,
    }


def set_paper_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_ate_forest(summaries, out_dir, timestamp, blue_ai, method):
    ordered = sorted(summaries, key=lambda d: abs(d["effect"]))
    labels = [d["label"] for d in ordered]
    y = np.arange(len(ordered))
    effects = np.array([d["effect"] for d in ordered])
    lower = np.array([d["ci_low"] for d in ordered])
    upper = np.array([d["ci_high"] for d in ordered])
    xerr = np.vstack([effects - lower, upper - effects])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axvline(0, color="#5F6368", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.axvspan(min(lower.min(), effects.min()) - 10, 0, color="#CC79A7", alpha=0.055, zorder=0)
    ax.errorbar(
        effects,
        y,
        xerr=xerr,
        fmt="o",
        color=AI_COLORS.get(blue_ai, "#333333"),
        ecolor="#4D4D4D",
        elinewidth=1.0,
        capsize=2.5,
        markersize=4.8,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("ATE: adversarial level minus benign level (Blue score)")
    ax.set_title("Figure 3. Causal intervention effects for Qwen-LLM", loc="left", pad=16)
    subtitle = (
        f"GP surrogate fitted on {AI_LABELS.get(blue_ai, blue_ai)} "
        f"with {method.replace('_', ' ').title()} observations"
    )
    ax.text(
        0.0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#666666",
        fontsize=8,
    )
    ax.text(
        0.005,
        -0.13,
        "More negative values indicate stronger causal degradation of Blue performance.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#666666",
        fontsize=7.5,
    )

    for yi, d in enumerate(ordered):
        txt = f"{d['from']} -> {d['to']}"
        x_text = d["effect"] - 6 if d["effect"] < 0 else d["effect"] + 6
        ha = "right" if d["effect"] < 0 else "left"
        ax.text(
            x_text,
            yi,
            txt,
            ha=ha,
            va="center",
            fontsize=7,
            color="#6A6A6A",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
        )

    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.12)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.18)
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"figure3_llm_ate_forest_{timestamp}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_radar(profile_by_ai, out_dir, timestamp):
    names = config.VAR_NAMES
    labels = [VAR_LABELS.get(n, n) for n in names]
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(6.4, 5.8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#777777", fontsize=7)
    ax.tick_params(axis="x", pad=6, labelsize=8)
    ax.spines["polar"].set_color("#BBBBBB")
    ax.spines["polar"].set_linewidth(0.8)
    ax.grid(True, alpha=0.24)

    for ai, values in profile_by_ai.items():
        vals = np.asarray([values[n] for n in names], dtype=float)
        vals = vals / max(float(np.max(vals)), 1e-9)
        vals = np.concatenate([vals, vals[:1]])
        color = AI_COLORS.get(ai, None)
        ax.plot(
            angles,
            vals,
            lw=1.9,
            color=color,
            marker="o",
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=AI_LABELS.get(ai, ai),
        )
        ax.fill(angles, vals, color=color, alpha=0.055)

    ax.set_title("Figure 4. Vulnerability profiles across Blue AI policies", pad=18, fontsize=10)
    ax.text(
        0.5,
        -0.10,
        "Each profile is normalized within AI; larger radius indicates stronger marginal sensitivity.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="#666666",
        fontsize=7.5,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.5,
    )

    fig.subplots_adjust(left=0.06, right=0.96, top=0.88, bottom=0.19)
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"figure4_ai_vulnerability_radar_{timestamp}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def save_summary_csv(llm_summaries, profile_by_ai, out_dir, timestamp):
    ate_path = os.path.join(out_dir, f"figure3_llm_ate_values_{timestamp}.csv")
    with open(ate_path, "w", encoding="utf-8") as f:
        f.write("variable,label,least_adversarial,most_adversarial,ate,ci_low,ci_high,range\n")
        for d in sorted(llm_summaries, key=lambda row: abs(row["effect"]), reverse=True):
            f.write(
                f"{d['variable']},{d['label']},{d['from']},{d['to']},"
                f"{d['effect']:.6f},{d['ci_low']:.6f},{d['ci_high']:.6f},{d['range']:.6f}\n"
            )

    radar_path = os.path.join(out_dir, f"figure4_vulnerability_profile_values_{timestamp}.csv")
    with open(radar_path, "w", encoding="utf-8") as f:
        f.write("blue_ai,variable,raw_sensitivity,normalized_sensitivity\n")
        for ai, values in profile_by_ai.items():
            max_v = max(max(values.values()), 1e-9)
            for name in config.VAR_NAMES:
                f.write(f"{ai},{name},{values[name]:.6f},{values[name] / max_v:.6f}\n")
    return [ate_path, radar_path]


def main():
    parser = argparse.ArgumentParser(description="Generate supplementary causal figures.")
    parser.add_argument("--benchmark", required=True, help="benchmark JSON file")
    parser.add_argument("--out_dir", default="figures", help="output directory")
    parser.add_argument("--timestamp", default=None, help="optional timestamp suffix")
    parser.add_argument("--llm_ai", default="llm-qwen", help="LLM Blue AI name")
    parser.add_argument("--llm_method", default="causal_bo", help="method used for Figure 3")
    parser.add_argument("--ais", nargs="+", default=["agg", "llm-qwen", "pass-agg"])
    parser.add_argument("--reference_samples", type=int, default=1200)
    args = parser.parse_args()

    set_paper_style()
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    runs, payload = load_runs(args.benchmark)
    rng = np.random.RandomState(2026)
    reference = generate_reference_samples(args.reference_samples, rng=rng)

    X_llm, y_llm = collect_training_data(runs, args.llm_ai, method=args.llm_method)
    gp_llm = fit_gp(X_llm, y_llm, causal=True)
    llm_summaries = [
        variable_effect_summary(gp_llm, name, reference)
        for name in config.VAR_NAMES
    ]

    profile_by_ai = {}
    for ai in args.ais:
        X_ai, y_ai = collect_training_data(runs, ai, method=None)
        gp_ai = fit_gp(X_ai, y_ai, causal=True)
        summaries = [variable_effect_summary(gp_ai, name, reference) for name in config.VAR_NAMES]
        profile_by_ai[ai] = {d["variable"]: d["range"] for d in summaries}

    saved = []
    saved.extend(plot_ate_forest(llm_summaries, args.out_dir, timestamp, args.llm_ai, args.llm_method))
    saved.extend(plot_radar(profile_by_ai, args.out_dir, timestamp))
    saved.extend(save_summary_csv(llm_summaries, profile_by_ai, args.out_dir, timestamp))

    print("Generated supplementary figures from:")
    print(f"  {args.benchmark}")
    print("Saved files:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
