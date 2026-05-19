"""
Acquisition functions and candidate generation for single-level CBO.

The objective is minimization:

    x* = argmin_x E[y | do(x)]
"""

import numpy as np
from scipy.stats import norm

import config
from atlatl_evaluator import N_DIM, vector_to_vars, vars_to_vector


def expected_improvement_min(gp, x_candidates, y_observed, xi=0.01):
    """
    Expected improvement for minimization.

    EI(x) = E[max(y_best - Y(x) - xi, 0)]
    """
    mu, var = gp.predict(x_candidates)
    sigma = np.sqrt(np.maximum(var, 1e-12))
    y_best = float(np.min(y_observed))
    improvement = y_best - mu - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def lower_confidence_bound_min(gp, x_candidates, kappa=2.0):
    """LCB acquisition for minimization: lower is better."""
    mu, var = gp.predict(x_candidates)
    sigma = np.sqrt(np.maximum(var, 1e-12))
    return mu - kappa * sigma


def select_ei_candidate(gp, x_candidates, y_observed, xi=0.01, lcb_kappa=2.0):
    ei = expected_improvement_min(gp, x_candidates, y_observed, xi=xi)
    max_ei = float(np.max(ei))
    if np.isfinite(max_ei) and max_ei > 1e-10:
        idx = int(np.argmax(ei))
        return idx, ei, "EI"

    lcb = lower_confidence_bound_min(gp, x_candidates, kappa=lcb_kappa)
    idx = int(np.argmin(lcb))
    # Return a positive score for display while preserving the old shape.
    return idx, -lcb, "LCB"


def _sample_terrain_probs(rng):
    """Sample terrain probabilities subject to p_urban+p_rough+p_marsh <= 1."""
    for _ in range(100):
        p_urban = rng.uniform(0.0, 0.5)
        p_rough = rng.uniform(0.0, 0.5)
        p_marsh = rng.uniform(0.0, 0.3)
        if p_urban + p_rough + p_marsh <= 1.0:
            return p_urban, p_rough, p_marsh
    vals = np.array([
        rng.uniform(0.0, 0.5),
        rng.uniform(0.0, 0.5),
        rng.uniform(0.0, 0.3),
    ])
    vals = vals / max(vals.sum(), 1.0)
    return vals.tolist()


def generate_x_candidates(n_samples, rng=None):
    """Generate mixed discrete/continuous scenario candidates."""
    if rng is None:
        rng = np.random.RandomState()

    candidates = np.zeros((n_samples, N_DIM), dtype=np.float64)
    for i in range(n_samples):
        x_vars = {
            "n_blue": int(rng.randint(1, 5)),
            "blue_side": config.SIDE_CATEGORIES[int(rng.randint(0, 4))],
            "blue_unit_type": config.UNIT_TYPE_CATEGORIES[int(rng.randint(0, 4))],
            "n_red": int(rng.randint(1, 5)),
            "red_ai": config.AI_CATEGORIES[int(rng.randint(0, len(config.AI_CATEGORIES)))],
            "max_phases": int(rng.randint(6, 21)),
            "red_unit_type": config.UNIT_TYPE_CATEGORIES[int(rng.randint(0, 4))],
        }
        p_urban, p_rough, p_marsh = _sample_terrain_probs(rng)
        x_vars["p_urban"] = p_urban
        x_vars["p_rough"] = p_rough
        x_vars["p_marsh"] = p_marsh
        candidates[i] = vars_to_vector(x_vars)
    return candidates


def generate_reference_samples(n_samples, rng=None):
    """Monte Carlo reference distribution for intervention analysis."""
    return generate_x_candidates(n_samples, rng=rng)


# Backward-compatible aliases for older imports.
minimax_acquisition = None
generate_d_candidates = generate_x_candidates
generate_u_candidates = generate_x_candidates
