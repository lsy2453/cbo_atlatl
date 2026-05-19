"""
Single-level causal Bayesian optimization for adversarial Atlatl scenarios.

Objective:

    x* = argmin_x E[y | do(x)]

where x is the 10-dimensional scenario configuration and y is the Blue AI
score. Lower y means a more damaging adversarial scenario.
"""

import json
import time

import numpy as np

import config
from acquisition import generate_reference_samples, generate_x_candidates, select_ei_candidate
from atlatl_evaluator import evaluate, vars_to_vector, vector_to_vars
from causal_gp import CausalGP, RBFKernel
from causal_graph import CausalDAG


class CBOOptimizer:
    def __init__(self, n_initial=20, n_iterations=30, n_candidates=200,
                 n_eval_seeds=3, seed=42):
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_candidates = n_candidates
        self.n_eval_seeds = n_eval_seeds
        self.rng = np.random.RandomState(seed)

        self.dag = CausalDAG()
        self.gp = CausalGP(RBFKernel(lengthscale=self._default_lengthscales()))

        self.X_observed = []
        self.y_observed = []
        self.x_history = []
        self.eval_history = []

        self.best_x = None
        self.best_y = float("inf")

        self.reference_samples = generate_reference_samples(300, self.rng)

    def _default_lengthscales(self):
        """Reasonable scales for the encoded mixed variable space."""
        ls = np.ones(len(config.VAR_NAMES), dtype=np.float64)
        for name in ["n_blue", "n_red"]:
            ls[config.VAR_INDICES[name]] = 1.0
        for name in ["blue_side", "blue_unit_type", "red_ai", "red_unit_type"]:
            ls[config.VAR_INDICES[name]] = 1.2
        ls[config.VAR_INDICES["max_phases"]] = 4.0
        for name in ["p_urban", "p_rough", "p_marsh"]:
            ls[config.VAR_INDICES[name]] = 0.2
        return ls

    def _format_x(self, x_vars):
        return (
            f"B={x_vars['n_blue']} {x_vars['blue_unit_type']}@{x_vars['blue_side']} | "
            f"R={x_vars['n_red']} {x_vars['red_unit_type']} {x_vars['red_ai']} | "
            f"ph={x_vars['max_phases']} | "
            f"terr=({x_vars['p_urban']:.2f},{x_vars['p_rough']:.2f},"
            f"{x_vars['p_marsh']:.2f})"
        )

    def _evaluate_point(self, x_vec):
        x_vars = vector_to_vars(x_vec)
        result = evaluate(x_vars, n_seeds=self.n_eval_seeds)
        score = result["mean"]

        encoded = vars_to_vector(x_vars)
        self.X_observed.append(encoded)
        self.y_observed.append(score)
        self.x_history.append(x_vars)
        self.eval_history.append(result)

        if score < self.best_y:
            self.best_y = score
            self.best_x = x_vars

        return x_vars, score, result

    def _initial_sampling(self):
        print(f"\n{'=' * 60}")
        print(f"Phase 1: Initial random interventions ({self.n_initial})")
        print(f"{'=' * 60}")

        samples = generate_x_candidates(self.n_initial, self.rng)
        for i, x_vec in enumerate(samples):
            x_vars, score, result = self._evaluate_point(x_vec)
            print(
                f"  [{i + 1:2d}/{self.n_initial}] "
                f"{self._format_x(x_vars)} | y={score:7.1f} "
                f"(sd={result['std']:.1f})"
            )

    def _cbo_iteration(self, iteration):
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)

        candidates = generate_x_candidates(self.n_candidates, self.rng)
        idx, acq_values, acq_name = select_ei_candidate(
            self.gp, candidates, self.y_observed
        )
        x_vars, score, result = self._evaluate_point(candidates[idx])

        print(
            f"  iter {iteration:2d}: {self._format_x(x_vars)} | "
            f"y={score:7.1f} | {acq_name}={acq_values[idx]:.3f} | "
            f"best={self.best_y:.1f}"
        )

    def optimize(self):
        start_time = time.time()

        self._initial_sampling()

        print(f"\nAfter initial sampling:")
        print(f"  x_best = {self.best_x}")
        print(f"  y_best = {self.best_y:.1f}")

        print(f"\n{'=' * 60}")
        print(f"Phase 2: Single-level CBO ({self.n_iterations} EI rounds)")
        print(f"{'=' * 60}")
        for i in range(self.n_iterations):
            self._cbo_iteration(i + 1)

        self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))

        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"COMPLETE ({elapsed:.1f}s, {len(self.y_observed)} evaluations)")
        print(f"{'=' * 60}")
        print(f"  x* = {self.best_x}")
        print(f"  min observed score = {self.best_y:.1f}")

        self._causal_explanation()
        return self.get_results()

    def _causal_explanation(self):
        """Generate post-hoc causal intervention summaries from the GP."""
        if self.best_x is None:
            return

        print(f"\n{'=' * 60}")
        print("Causal Analysis")
        print(f"{'=' * 60}")

        x_best_vec = vars_to_vector(self.best_x)
        mu_best, var_best = self.gp.predict(x_best_vec.reshape(1, -1))
        print(f"\nPredicted E[y | do(x*)] = {mu_best[0]:.1f} "
              f"+/- {np.sqrt(var_best[0]):.1f}")

        print("\nInterventional marginals at variable extremes/categories:")
        for name in config.VAR_NAMES:
            idx = config.VAR_INDICES[name]
            spec = config.SCENARIO_VARS[name]
            summaries = []
            if spec["type"] == "categorical":
                for j, cat in enumerate(spec["categories"]):
                    mean, _ = self.gp.do_marginal(idx, j, self.reference_samples)
                    summaries.append((cat, mean))
            elif spec["type"] == "integer":
                for value in [spec["low"], spec["high"]]:
                    mean, _ = self.gp.do_marginal(idx, value, self.reference_samples)
                    summaries.append((value, mean))
            else:
                for value in [spec["low"], spec["high"]]:
                    mean, _ = self.gp.do_marginal(idx, value, self.reference_samples)
                    summaries.append((f"{value:.2f}", mean))
            best_level, best_mean = min(summaries, key=lambda item: item[1])
            print(f"  {name:15s}: most adversarial {best_level} "
                  f"(E[y]={best_mean:.1f})")

        print("\nATE examples relative to low/baseline levels:")
        ate_specs = [
            ("blue_unit_type", "infantry", "artillery"),
            ("red_unit_type", "infantry", "artillery"),
            ("red_ai", "passive", "agg"),
            ("n_blue", 1, 4),
            ("n_red", 1, 4),
            ("max_phases", 6, 20),
        ]
        for name, value_from, value_to in ate_specs:
            idx = config.VAR_INDICES[name]
            spec = config.SCENARIO_VARS[name]
            if spec["type"] == "categorical":
                value_from_idx = spec["categories"].index(value_from)
                value_to_idx = spec["categories"].index(value_to)
            else:
                value_from_idx = value_from
                value_to_idx = value_to
            ate = self.gp.causal_ate(
                idx, value_from_idx, value_to_idx, self.reference_samples
            )
            print(f"  do({name}: {value_from} -> {value_to}): ATE={ate:+.1f}")

        print("\nActive causal paths from x* variables:")
        for var in [
            "blue_unit_type", "red_unit_type", "red_ai",
            "p_urban", "p_marsh", "max_phases",
        ]:
            paths = self.dag.get_causal_path(var, "y")
            for path in paths[:2]:
                print(f"  {' -> '.join(path)}")

    def get_results(self):
        return {
            "best_x": self.best_x,
            "best_y": float(self.best_y),
            "n_evaluations": len(self.y_observed),
            "n_initial": self.n_initial,
            "n_iterations": self.n_iterations,
            "n_candidates": self.n_candidates,
            "blue_ai": config.BLUE_AI,
            "all_scores": [float(s) for s in self.y_observed],
            "all_x": self.x_history,
        }

    def save_results(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.get_results(), f, indent=2)
        print(f"Results saved to {filepath}")
