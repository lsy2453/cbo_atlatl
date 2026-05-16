"""
Causal acquisition function for minimax CBO with do-calculus.

Problem: max_d min_u E[y | do(d), do(u)]

Key fixes from previous version:
1. Uses do_d_worst_u for inner minimization (proper do-calculus)
2. Acquisition compares robust values across d candidates (not against
   an inflated global best)
3. Exploration term is pathway-specific uncertainty, not generic variance
"""

import numpy as np
from scipy.stats import norm
from atlatl_evaluator import D_INDICES, U_INDICES


def minimax_acquisition(gp, d_candidates, u_candidates, y_observed,
                         dag=None, beta_inner=1.5, beta_explore=0.5):
    """
    Minimax CBO acquisition with do-calculus.

    For each candidate d_i:
      1. Inner loop: find worst u via do_d_worst_u (adversarial do-calculus)
         u*_i = argmin_u  LCB[y | do(d_i), do(u)]
      2. Compute robust value: R(d_i) = LCB[y | do(d_i), do(u*_i)]

    Then select d with highest: R(d) + beta_explore * sigma(d, u*)
    This balances exploitation (high robust value) and exploration
    (high uncertainty at the worst-case point).

    Returns
    -------
    best_d_idx, best_u_idx, acq_values_for_all_d
    """
    n_d = len(d_candidates)
    robust_values = np.zeros(n_d)
    worst_u_indices = np.zeros(n_d, dtype=int)
    uncertainties = np.zeros(n_d)

    dim = len(D_INDICES) + len(U_INDICES)

    for i in range(n_d):
        # do-calculus: min_u E[y | do(d_i), do(u)]
        worst_idx, worst_lcb, all_lcb = gp.do_d_worst_u(
            d_candidates[i], u_candidates, beta=beta_inner
        )
        worst_u_indices[i] = worst_idx
        robust_values[i] = worst_lcb

        # Uncertainty at the worst-case point (for exploration)
        x_worst = np.zeros(dim)
        x_worst[D_INDICES] = d_candidates[i]
        x_worst[U_INDICES] = u_candidates[worst_idx]
        _, var = gp.predict(x_worst.reshape(1, -1))
        uncertainties[i] = np.sqrt(var[0])

    # Combined acquisition: robust value + exploration bonus
    # Normalize both to [0, 1]
    r_min, r_max = robust_values.min(), robust_values.max()
    if r_max > r_min:
        r_norm = (robust_values - r_min) / (r_max - r_min)
    else:
        r_norm = np.zeros(n_d)

    u_min, u_max = uncertainties.min(), uncertainties.max()
    if u_max > u_min:
        u_norm = (uncertainties - u_min) / (u_max - u_min)
    else:
        u_norm = np.zeros(n_d)

    acq = (1 - beta_explore) * r_norm + beta_explore * u_norm

    best_d_idx = np.argmax(acq)
    best_u_idx = worst_u_indices[best_d_idx]

    return int(best_d_idx), int(best_u_idx), acq


def generate_d_candidates(n_samples, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    c = np.zeros((n_samples, len(D_INDICES)))
    c[:, 0] = rng.randint(1, 5, n_samples)  # n_blue: 1-4
    c[:, 1] = rng.randint(0, 4, n_samples)  # blue_side: 0-3
    return c


def generate_u_candidates(n_samples, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    c = np.zeros((n_samples, len(U_INDICES)))
    c[:, 0] = rng.randint(1, 5, n_samples)     # n_red
    c[:, 1] = rng.randint(0, 4, n_samples)     # red_ai
    c[:, 2] = rng.randint(6, 21, n_samples)    # max_phases
    c[:, 3] = rng.uniform(0, 0.5, n_samples)   # p_urban
    c[:, 4] = rng.uniform(0, 0.5, n_samples)   # p_rough
    c[:, 5] = rng.uniform(0, 0.3, n_samples)   # p_marsh
    c[:, 6] = rng.uniform(0, 1, n_samples)     # seed
    return c
