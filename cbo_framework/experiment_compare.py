"""
Comparative CBO experiment across three AI types.
Tests whether causal decomposition reveals different vulnerability patterns.

AI types:
  1. pass-agg (rule-based): deterministic heuristic
  2. llm (LLM-mock): simulates LLM decision patterns
  3. passive (baseline): minimal behavior

For each AI, CBO finds:
  - d* (robust configuration)
  - Vulnerability causal decomposition
  - ATE of key variables
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                "..", "atlatl-public-master", "server"))

import numpy as np
import json
from cbo_optimizer import CBOOptimizer
import config


def run_experiment(blue_ai, n_initial=8, n_iter=10, label=""):
    print(f"\n{'#'*60}")
    print(f"# CBO for Blue AI: {blue_ai} {label}")
    print(f"{'#'*60}")

    config.BLUE_AI = blue_ai

    optimizer = CBOOptimizer(
        n_initial=n_initial,
        n_iterations=n_iter,
        n_d_candidates=30,
        n_u_candidates=50,
        n_eval_seeds=1,
        seed=42,
    )
    results = optimizer.optimize()
    results["blue_ai"] = blue_ai
    return results, optimizer


if __name__ == "__main__":
    all_results = {}

    # Test three AI types
    for ai_name in ["pass-agg", "llm", "passive"]:
        results, optimizer = run_experiment(ai_name, n_initial=6, n_iter=8)
        all_results[ai_name] = results

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"COMPARATIVE SUMMARY")
    print(f"{'='*60}")
    print(f"{'AI':<12} {'d*':>20} {'Robust Value':>14} {'Evals':>6}")
    print(f"{'-'*55}")
    for ai_name, r in all_results.items():
        d_str = f"n={r['best_d']['n_blue']},{r['best_d']['blue_side']}"
        print(f"{ai_name:<12} {d_str:>20} {r['robust_value']:>14.1f} "
              f"{r['n_evaluations']:>6}")

    # Save
    output = {k: v for k, v in all_results.items()}
    with open("comparison_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to comparison_results.json")
