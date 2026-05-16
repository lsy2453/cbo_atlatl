"""
Causal Bayesian Optimization for minimax robustness.
Solves: d* = arg max_d  min_u  E[ y | do(d), do(u) ]

do-calculus integration:
- Each evaluation IS an intervention: we set do(d, u) and observe y
- GP models P(y | do(d, u)) via causal additive kernel
- Inner loop uses do_d_worst_u for adversarial search
- ATE analysis shows causal effect of changing d
- Per-pathway decomposition explains WHY d* is robust
"""

import numpy as np
import json
import time

from causal_graph import CausalDAG
from causal_gp import CausalGP, CausalKernel
from acquisition import (minimax_acquisition, generate_d_candidates,
                          generate_u_candidates)
from atlatl_evaluator import (evaluate, vars_to_vector, vector_to_vars,
                                D_INDICES, U_INDICES)


class CBOOptimizer:
    def __init__(self, n_initial=20, n_iterations=30,
                 n_d_candidates=50, n_u_candidates=100,
                 n_eval_seeds=3, seed=42):
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_d_candidates = n_d_candidates
        self.n_u_candidates = n_u_candidates
        self.n_eval_seeds = n_eval_seeds
        self.rng = np.random.RandomState(seed)

        self.dag = CausalDAG()
        self.gp = CausalGP(CausalKernel(self.dag))

        # Observation history: all (d, u) -> y pairs
        self.X_observed = []
        self.y_observed = []
        self.d_history = []
        self.u_history = []

        # Robust solution tracking
        self.best_robust_value = float('-inf')
        self.best_d = None
        self.best_worst_u = None

        # Reference u samples for do-calculus marginals
        self.u_reference = generate_u_candidates(200, self.rng)

    def _initial_sampling(self):
        print(f"\n{'='*60}")
        print(f"Phase 1: Initial sampling ({self.n_initial} evaluations)")
        print(f"{'='*60}")

        d_samples = generate_d_candidates(self.n_initial, self.rng)
        u_samples = generate_u_candidates(self.n_initial, self.rng)

        for i in range(self.n_initial):
            d_vars, u_vars, score = self._evaluate_point(
                d_samples[i], u_samples[i])
            print(f"  [{i+1}/{self.n_initial}] "
                  f"d=(blue={d_vars['n_blue']}, {d_vars['blue_side']}) | "
                  f"u=(red={u_vars['n_red']}, {u_vars['red_ai']}, "
                  f"ph={u_vars['max_phases']}) | "
                  f"y={score:.1f}")

    def _evaluate_point(self, d_vec, u_vec):
        """Run do(d, u) intervention in Atlatl and record result."""
        full_vec = np.zeros(len(D_INDICES) + len(U_INDICES))
        full_vec[D_INDICES] = d_vec
        full_vec[U_INDICES] = u_vec
        d_vars, u_vars = vector_to_vars(full_vec)

        result = evaluate(d_vars, u_vars, n_seeds=self.n_eval_seeds)
        score = result["mean"]

        x_vec = vars_to_vector(d_vars, u_vars)
        self.X_observed.append(x_vec)
        self.y_observed.append(score)
        self.d_history.append(d_vars)
        self.u_history.append(u_vars)

        return d_vars, u_vars, score

    def _update_robust_solution(self):
        """
        Proper robust value computation using do-calculus.
        For each unique d observed, compute min_u E[y|do(d),do(u)]
        using the GP's do_d_worst_u method over reference u samples.
        """
        # Get unique d configurations seen
        seen_d = {}
        for i, d in enumerate(self.d_history):
            key = f"{d['n_blue']}_{d['blue_side']}"
            if key not in seen_d:
                seen_d[key] = {
                    "d_vars": d,
                    "d_vec": np.array(self.X_observed[i])[D_INDICES],
                }

        # For each d, use GP to estimate worst-case u
        best_val = float('-inf')
        for key, info in seen_d.items():
            _, worst_lcb, _ = self.gp.do_d_worst_u(
                info["d_vec"], self.u_reference, beta=1.5)
            if worst_lcb > best_val:
                best_val = worst_lcb
                self.best_d = info["d_vars"]
                self.best_robust_value = worst_lcb
                # Find what u gives worst case
                worst_idx, _, _ = self.gp.do_d_worst_u(
                    info["d_vec"], self.u_reference, beta=1.5)
                worst_u_full = np.zeros(len(D_INDICES) + len(U_INDICES))
                worst_u_full[D_INDICES] = info["d_vec"]
                worst_u_full[U_INDICES] = self.u_reference[worst_idx]
                _, self.best_worst_u = vector_to_vars(worst_u_full)

    def _cbo_iteration(self, iteration):
        """Single CBO iteration with do-calculus."""
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)

        # Generate candidates
        d_cands = generate_d_candidates(self.n_d_candidates, self.rng)
        u_cands = generate_u_candidates(self.n_u_candidates, self.rng)

        # Minimax acquisition with do-calculus
        d_idx, u_idx, acq_values = minimax_acquisition(
            self.gp, d_cands, u_cands, self.y_observed, self.dag)
        acq_val = acq_values[d_idx]

        # Evaluate selected do(d, u) intervention
        d_vars, u_vars, score = self._evaluate_point(
            d_cands[d_idx], u_cands[u_idx])

        # Update robust solution using GP + do-calculus
        self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
        self._update_robust_solution()

        # Causal decomposition at this point
        x_vec = vars_to_vector(d_vars, u_vars)
        decomp = self.gp.causal_effect_decomposition(x_vec)

        print(f"  iter {iteration:2d}: "
              f"d=(blue={d_vars['n_blue']}, {d_vars['blue_side']:5s}) | "
              f"u=(red={u_vars['n_red']}, {u_vars['red_ai']:10s}, "
              f"ph={u_vars['max_phases']:2d}) | "
              f"y={score:7.1f} | acq={acq_val:.3f} | "
              f"robust={self.best_robust_value:.1f}")
        decomp_str = " | ".join(f"{k}={v:+.0f}" for k, v in
                                 sorted(decomp.items(),
                                        key=lambda x: abs(x[1]),
                                        reverse=True)[:3])
        print(f"           do-decomp: {decomp_str}")

    def optimize(self):
        start_time = time.time()

        # Phase 1: Initial sampling (interventional data collection)
        self._initial_sampling()
        self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
        self._update_robust_solution()
        print(f"\nAfter initial sampling (do-calculus robust estimate):")
        print(f"  Best d: {self.best_d} | robust value: {self.best_robust_value:.1f}")

        # Phase 2: CBO with do-calculus
        print(f"\n{'='*60}")
        print(f"Phase 2: CBO iterations ({self.n_iterations} rounds)")
        print(f"{'='*60}")
        for i in range(self.n_iterations):
            self._cbo_iteration(i + 1)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"COMPLETE ({elapsed:.1f}s, {len(self.y_observed)} evaluations)")
        print(f"{'='*60}")
        print(f"  d* = {self.best_d}")
        print(f"  worst u = {self.best_worst_u}")
        print(f"  robust value = {self.best_robust_value:.1f}")

        self._causal_explanation()
        return self.get_results()

    def _causal_explanation(self):
        """Generate causal explanations using do-calculus."""
        print(f"\n{'='*60}")
        print(f"Causal Explanation (do-calculus)")
        print(f"{'='*60}")

        if self.best_d is None:
            return

        # 1. Per-pathway decomposition at (d*, worst_u)
        x_best = vars_to_vector(self.best_d, self.best_worst_u)
        decomp = self.gp.causal_effect_decomposition(x_best)
        print(f"\nE[y | do(d*), do(worst_u)] decomposition:")
        total = sum(abs(v) for v in decomp.values())
        for name, val in sorted(decomp.items(), key=lambda x: abs(x[1]),
                                reverse=True):
            pct = abs(val) / (total + 1e-10) * 100
            print(f"  {name:20s}: {val:+8.1f} ({pct:.0f}%)")

        # 2. ATE: what happens if we change n_blue?
        print(f"\nAverage Treatment Effects (do-calculus):")
        d_base = np.array(x_best[D_INDICES])
        for n_blue_alt in [1, 2, 3, 4]:
            if n_blue_alt == self.best_d["n_blue"]:
                continue
            d_alt = d_base.copy()
            d_alt[0] = n_blue_alt
            ate, ate_pw = self.gp.causal_ate(d_base, d_alt, self.u_reference)
            top_pathway = max(ate_pw.items(), key=lambda x: abs(x[1]))
            print(f"  do(n_blue={self.best_d['n_blue']}→{n_blue_alt}): "
                  f"ATE={ate:+.1f}, "
                  f"main pathway: {top_pathway[0]} ({top_pathway[1]:+.1f})")

        # 3. Interventional marginal for best d
        d_best_vec = np.array(x_best[D_INDICES])
        mean_do, var_do = self.gp.do_d_marginal(d_best_vec, self.u_reference)
        print(f"\nInterventional marginal E[y | do(d*)]:")
        print(f"  mean = {mean_do:.1f}, std = {np.sqrt(var_do):.1f}")
        print(f"  (averaged over {len(self.u_reference)} u samples)")

        # 4. Causal paths
        print(f"\nActive causal paths:")
        for var in ["n_blue", "n_red", "red_ai", "p_urban", "max_phases"]:
            for path in self.dag.get_causal_path(var, "y")[:1]:
                print(f"  {' → '.join(path)}")

    def get_results(self):
        return {
            "best_d": self.best_d,
            "best_worst_u": self.best_worst_u,
            "robust_value": self.best_robust_value,
            "n_evaluations": len(self.y_observed),
            "all_scores": [float(s) for s in self.y_observed],
        }

    def save_results(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.get_results(), f, indent=2)
        print(f"Results saved to {filepath}")
