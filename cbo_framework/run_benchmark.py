"""
Run method x AI benchmark for adversarial Atlatl scenario search.

Example quick run:
    python cbo_framework/run_benchmark.py --blue_ais pass-agg agg pascal --methods random_search causal_bo --budget 8 --n_initial 4 --seeds 1

Example full run with Qwen:
    python cbo_framework/run_benchmark.py --blue_ais pass-agg agg pascal llm-qwen --methods random_search standard_bo neighborhood sobol_guided causal_bo --budget 25 --n_initial 10 --seeds 1 2 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from benchmark_methods import METHODS, run_method


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark adversarial search methods across Blue AIs."
    )
    parser.add_argument("--blue_ais", nargs="+", required=True,
                        help="Blue AI policies, e.g. pass-agg agg pascal llm-qwen")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS,
                        help="search methods to compare")
    parser.add_argument("--budget", type=int, default=25,
                        help="total scenario evaluations per AI/method/seed")
    parser.add_argument("--n_initial", type=int, default=10,
                        help="initial random evaluations for BO/local methods")
    parser.add_argument("--n_candidates", type=int, default=50,
                        help="candidate scenarios per BO iteration")
    parser.add_argument("--n_eval_seeds", type=int, default=1,
                        help="Monte Carlo seeds per scenario evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="benchmark random seeds")
    parser.add_argument("--out_dir", default="benchmark_results",
                        help="output directory")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="record failures instead of stopping")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"benchmark_{timestamp}.json")

    all_runs = []
    start_all = time.time()
    for blue_ai in args.blue_ais:
        for method in args.methods:
            for seed in args.seeds:
                print(f"\n=== AI={blue_ai} | method={method} | seed={seed} ===")
                try:
                    result = run_method(
                        method=method,
                        blue_ai=blue_ai,
                        budget=args.budget,
                        n_initial=args.n_initial,
                        n_candidates=args.n_candidates,
                        n_eval_seeds=args.n_eval_seeds,
                        seed=seed,
                    )
                    print(
                        f"best_y={result['best_y']:.1f} "
                        f"evals={result['n_evaluations']} "
                        f"elapsed={result['elapsed_sec']:.1f}s"
                    )
                    all_runs.append(result)
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    print(f"FAILED: {exc}")
                    all_runs.append({
                        "method": method,
                        "blue_ai": blue_ai,
                        "seed": seed,
                        "status": "failed",
                        "error": str(exc),
                    })

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "created_at": timestamp,
                        "elapsed_sec": time.time() - start_all,
                        "config": vars(args),
                        "runs": all_runs,
                    }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved benchmark results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
