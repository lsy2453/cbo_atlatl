"""
Compare adversarial CBO results across fixed Blue AI policies.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "atlatl-public-master", "server"))

import config
from cbo_optimizer import CBOOptimizer


def run_experiment(blue_ai, n_initial=8, n_iter=10, label=""):
    print(f"\n{'#' * 60}")
    print(f"# Single-level CBO for Blue AI: {blue_ai} {label}")
    print(f"{'#' * 60}")

    config.BLUE_AI = blue_ai
    optimizer = CBOOptimizer(
        n_initial=n_initial,
        n_iterations=n_iter,
        n_candidates=80,
        n_eval_seeds=1,
        seed=42,
    )
    results = optimizer.optimize()
    results["blue_ai"] = blue_ai
    return results, optimizer


if __name__ == "__main__":
    all_results = {}
    for ai_name in ["pass-agg", "llm", "passive"]:
        results, _ = run_experiment(ai_name, n_initial=6, n_iter=8)
        all_results[ai_name] = results

    print(f"\n{'=' * 60}")
    print("COMPARATIVE SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'AI':<12} {'Best scenario':>48} {'Min score':>12} {'Evals':>6}")
    print(f"{'-' * 82}")
    for ai_name, r in all_results.items():
        x = r["best_x"]
        x_str = (
            f"B{x['n_blue']}-{x['blue_unit_type']}@{x['blue_side']} / "
            f"R{x['n_red']}-{x['red_unit_type']} {x['red_ai']}"
        )
        print(f"{ai_name:<12} {x_str:>48} {r['best_y']:>12.1f} "
              f"{r['n_evaluations']:>6}")

    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved to comparison_results.json")
