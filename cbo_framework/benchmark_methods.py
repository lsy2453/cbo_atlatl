"""
Benchmark adversarial scenario-search methods under a common budget.

Methods implemented:
- random_search
- standard_bo
- causal_bo
- neighborhood
- sobol_guided

The objective for all methods is identical:

    minimize E[y | do(x)]

where y is the Blue AI score. Lower is more adversarial.
"""

import time

import numpy as np

import config
from acquisition import generate_x_candidates, select_ei_candidate
from atlatl_evaluator import evaluate, vars_to_vector, vector_to_vars
from causal_gp import CausalGP, RBFKernel


METHODS = [
    "random_search",
    "standard_bo",
    "causal_bo",
    "neighborhood",
    "sobol_guided",
]


def default_causal_lengthscales():
    ls = np.ones(len(config.VAR_NAMES), dtype=np.float64)
    for name in ["n_blue", "n_red"]:
        ls[config.VAR_INDICES[name]] = 1.0
    for name in ["blue_side", "blue_unit_type", "red_ai", "red_unit_type"]:
        ls[config.VAR_INDICES[name]] = 1.2
    ls[config.VAR_INDICES["max_phases"]] = 4.0
    for name in ["p_urban", "p_rough", "p_marsh"]:
        ls[config.VAR_INDICES[name]] = 0.2
    return ls


class BudgetedSearch:
    def __init__(self, blue_ai, budget, n_eval_seeds, seed):
        self.blue_ai = blue_ai
        self.budget = budget
        self.n_eval_seeds = n_eval_seeds
        self.rng = np.random.RandomState(seed)
        self.X = []
        self.scores = []
        self.x_vars = []
        self.best_x = None
        self.best_y = float("inf")

    @property
    def remaining(self):
        return self.budget - len(self.scores)

    def evaluate_vec(self, x_vec):
        if self.remaining <= 0:
            raise RuntimeError("Evaluation budget exhausted.")
        old_blue_ai = config.BLUE_AI
        config.BLUE_AI = self.blue_ai
        try:
            x = vector_to_vars(x_vec)
            result = evaluate(x, n_seeds=self.n_eval_seeds)
        finally:
            config.BLUE_AI = old_blue_ai
        score = float(result["mean"])
        encoded = vars_to_vector(x)
        self.X.append(encoded)
        self.scores.append(score)
        self.x_vars.append(x)
        if score < self.best_y:
            self.best_y = score
            self.best_x = x
        return x, score

    def evaluate_random(self, n):
        candidates = generate_x_candidates(min(n, self.remaining), self.rng)
        for x_vec in candidates:
            self.evaluate_vec(x_vec)

    def result(self, method, seed, elapsed, extra=None):
        payload = {
            "method": method,
            "blue_ai": self.blue_ai,
            "seed": seed,
            "budget": self.budget,
            "n_eval_seeds": self.n_eval_seeds,
            "best_x": self.best_x,
            "best_y": float(self.best_y),
            "n_evaluations": len(self.scores),
            "all_scores": [float(s) for s in self.scores],
            "all_x": self.x_vars,
            "elapsed_sec": float(elapsed),
        }
        if extra:
            payload.update(extra)
        return payload


def run_random_search(blue_ai, budget, n_eval_seeds=1, seed=42, **kwargs):
    start = time.time()
    search = BudgetedSearch(blue_ai, budget, n_eval_seeds, seed)
    search.evaluate_random(budget)
    return search.result("random_search", seed, time.time() - start)


def _run_bo(blue_ai, budget, n_initial, n_candidates, n_eval_seeds, seed,
            method, lengthscale):
    start = time.time()
    search = BudgetedSearch(blue_ai, budget, n_eval_seeds, seed)
    search.evaluate_random(min(n_initial, budget))
    gp = CausalGP(RBFKernel(lengthscale=lengthscale))

    while search.remaining > 0:
        gp.fit(np.asarray(search.X), np.asarray(search.scores))
        candidates = generate_x_candidates(n_candidates, search.rng)
        idx, _, _ = select_ei_candidate(gp, candidates, search.scores)
        search.evaluate_vec(candidates[idx])

    return search.result(method, seed, time.time() - start, {
        "n_initial": n_initial,
        "n_candidates": n_candidates,
    })


def run_standard_bo(blue_ai, budget, n_initial=10, n_candidates=100,
                    n_eval_seeds=1, seed=42, **kwargs):
    return _run_bo(
        blue_ai=blue_ai,
        budget=budget,
        n_initial=n_initial,
        n_candidates=n_candidates,
        n_eval_seeds=n_eval_seeds,
        seed=seed,
        method="standard_bo",
        lengthscale=np.ones(len(config.VAR_NAMES), dtype=np.float64),
    )


def run_causal_bo(blue_ai, budget, n_initial=10, n_candidates=100,
                  n_eval_seeds=1, seed=42, **kwargs):
    return _run_bo(
        blue_ai=blue_ai,
        budget=budget,
        n_initial=n_initial,
        n_candidates=n_candidates,
        n_eval_seeds=n_eval_seeds,
        seed=seed,
        method="causal_bo",
        lengthscale=default_causal_lengthscales(),
    )


def _candidate_values(name, current, step_scale=0.15):
    spec = config.SCENARIO_VARS[name]
    if spec["type"] == "categorical":
        return [i for i in range(len(spec["categories"])) if i != int(round(current))]
    if spec["type"] == "integer":
        vals = [int(round(current)) - 1, int(round(current)) + 1]
        return sorted({v for v in vals if spec["low"] <= v <= spec["high"]})
    span = spec["high"] - spec["low"]
    vals = [current - step_scale * span, current + step_scale * span]
    return [float(np.clip(v, spec["low"], spec["high"])) for v in vals]


def _neighbors(x_vec, variables=None):
    variables = variables or config.VAR_NAMES
    for name in variables:
        idx = config.VAR_INDICES[name]
        for value in _candidate_values(name, x_vec[idx]):
            child = np.array(x_vec, dtype=np.float64, copy=True)
            child[idx] = value
            yield child


def run_neighborhood(blue_ai, budget, n_initial=10, top_k=3,
                     n_eval_seeds=1, seed=42, **kwargs):
    start = time.time()
    search = BudgetedSearch(blue_ai, budget, n_eval_seeds, seed)
    search.evaluate_random(min(n_initial, budget))

    starts = np.argsort(search.scores)[:max(1, min(top_k, len(search.scores)))]
    queue = [np.array(search.X[i], copy=True) for i in starts]

    while search.remaining > 0 and queue:
        current = queue.pop(0)
        current_score = min(search.scores)
        improved = False
        for child in _neighbors(current):
            if search.remaining <= 0:
                break
            _, score = search.evaluate_vec(child)
            if score < current_score:
                queue.append(vars_to_vector(search.best_x))
                improved = True
                break
        if not improved and search.remaining > 0 and not queue:
            queue.append(generate_x_candidates(1, search.rng)[0])

    return search.result("neighborhood", seed, time.time() - start, {
        "n_initial": n_initial,
        "top_k": top_k,
    })


def _variance_effect(scores, values, is_continuous):
    scores = np.asarray(scores, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.var(scores) < 1e-12:
        return 0.0
    if is_continuous:
        quantiles = np.quantile(values, [0.0, 1 / 3, 2 / 3, 1.0])
        groups = []
        for lo, hi in zip(quantiles[:-1], quantiles[1:]):
            mask = (values >= lo) & (values <= hi)
            if np.any(mask):
                groups.append(scores[mask])
    else:
        groups = [scores[values == v] for v in np.unique(values)]
    grand = np.mean(scores)
    between = 0.0
    for group in groups:
        if len(group):
            between += len(group) * (np.mean(group) - grand) ** 2
    return float(between / (len(scores) * np.var(scores)))


def estimate_sobol_like_importance(X, scores):
    """Lightweight first-order variance attribution used when SALib is absent."""
    effects = {}
    X = np.asarray(X, dtype=float)
    for name in config.VAR_NAMES:
        idx = config.VAR_INDICES[name]
        spec = config.SCENARIO_VARS[name]
        effects[name] = _variance_effect(
            scores, X[:, idx], is_continuous=(spec["type"] == "continuous")
        )
    return effects


def run_sobol_guided(blue_ai, budget, n_initial=16, top_m=4,
                     n_eval_seeds=1, seed=42, **kwargs):
    start = time.time()
    search = BudgetedSearch(blue_ai, budget, n_eval_seeds, seed)
    search.evaluate_random(min(n_initial, budget))

    effects = estimate_sobol_like_importance(search.X, search.scores)
    ranked_vars = [
        name for name, _ in sorted(effects.items(), key=lambda item: item[1],
                                  reverse=True)
    ][:top_m]

    current = vars_to_vector(search.best_x)
    while search.remaining > 0:
        children = list(_neighbors(current, variables=ranked_vars))
        if not children:
            break
        best_child = None
        best_score = float("inf")
        for child in children:
            if search.remaining <= 0:
                break
            _, score = search.evaluate_vec(child)
            if score < best_score:
                best_score = score
                best_child = child
        if best_child is not None and best_score <= search.best_y:
            current = vars_to_vector(search.best_x)
        else:
            current = generate_x_candidates(1, search.rng)[0]

    return search.result("sobol_guided", seed, time.time() - start, {
        "n_initial": n_initial,
        "top_m": top_m,
        "sobol_like_importance": effects,
        "guided_variables": ranked_vars,
    })


RUNNERS = {
    "random_search": run_random_search,
    "standard_bo": run_standard_bo,
    "causal_bo": run_causal_bo,
    "neighborhood": run_neighborhood,
    "sobol_guided": run_sobol_guided,
}


def run_method(method, **kwargs):
    if method not in RUNNERS:
        raise ValueError(f"Unknown method {method}. Valid methods: {sorted(RUNNERS)}")
    return RUNNERS[method](**kwargs)
