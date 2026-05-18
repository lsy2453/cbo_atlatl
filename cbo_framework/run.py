"""
Entry point for single-level causal Bayesian optimization on Atlatl.

Usage:
    python run.py --n_initial 10 --n_iter 15 --n_candidates 100
"""

import os
import sys


ATLATL_SERVER = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "atlatl-public-master", "server")
)

if ATLATL_SERVER not in sys.path:
    sys.path.insert(0, ATLATL_SERVER)

if not os.path.exists(os.path.join(ATLATL_SERVER, "game.py")):
    print("Error: cannot find Atlatl server/game.py")
    print(f"Checked path: {ATLATL_SERVER}")
    sys.exit(1)

import config
from cbo_optimizer import CBOOptimizer

config.ATLATL_SERVER_PATH = ATLATL_SERVER


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Single-level CBO for adversarial Atlatl scenario search"
    )
    parser.add_argument("--n_initial", type=int, default=10,
                        help="number of initial random interventions")
    parser.add_argument("--n_iter", type=int, default=15,
                        help="number of CBO/EI iterations")
    parser.add_argument("--n_candidates", type=int, default=100,
                        help="candidate scenarios sampled per EI iteration")
    parser.add_argument("--n_seeds", type=int, default=2,
                        help="Monte Carlo seeds per scenario evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="optimizer random seed")
    parser.add_argument("--output", type=str, default="results.json",
                        help="output JSON file")
    args = parser.parse_args()

    print(f"Atlatl path: {ATLATL_SERVER}")
    print("\nConfiguration:")
    print(f"  initial samples: {args.n_initial}")
    print(f"  CBO iterations:  {args.n_iter}")
    print(f"  EI candidates:   {args.n_candidates}")
    print(f"  eval seeds:      {args.n_seeds}")
    print(f"  random seed:     {args.seed}")
    print(f"  fixed Blue AI:   {config.BLUE_AI}")

    optimizer = CBOOptimizer(
        n_initial=args.n_initial,
        n_iterations=args.n_iter,
        n_candidates=args.n_candidates,
        n_eval_seeds=args.n_seeds,
        seed=args.seed,
    )
    optimizer.optimize()
    optimizer.save_results(args.output)
