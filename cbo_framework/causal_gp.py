"""
Causal Gaussian Process with do-calculus.

Key do-calculus operations:
1. predict(x) = E[y | do(d,u)]  -- since d,u are DAG root nodes,
   truncated factorization gives P(y|do(d,u)) = P(y|d,u)
2. do_d_marginal(d, U) = E_u[y | do(d)] = (1/N) sum E[y|do(d,u_i)]
   -- interventional marginal, averages out adversarial uncertainty
3. do_d_worst_u(d, U) = min_u E[y | do(d), do(u)]
   -- adversarial do-calculus for minimax
4. causal_ate(d1, d2) = E[y|do(d2)] - E[y|do(d1)]
   -- average treatment effect of changing d
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, cho_factor
from causal_graph import CausalDAG
from atlatl_evaluator import (FORCE_INDICES, TERRAIN_INDICES, AI_INDEX,
                               PHASE_INDEX, D_INDICES, U_INDICES)


class CausalKernel:
    """Additive kernel decomposed along causal DAG pathways."""

    def __init__(self, dag=None):
        if dag is None:
            dag = CausalDAG()
        # Lengthscales (smaller = variable has stronger causal effect)
        self.ls_force = 1.0
        self.ls_terrain = 3.0
        self.ls_ai = 1.5
        self.ls_phase = 2.0
        self.ls_interact = 4.0
        # Signal variance per pathway
        self.var_force = 200.0
        self.var_terrain = 30.0
        self.var_ai = 150.0
        self.var_phase = 50.0
        self.var_interact = 50.0
        # Noise
        self.noise_var = 25.0

    def _rbf(self, X1, X2, ls, var, idx):
        d = cdist(X1[:, idx] / ls, X2[:, idx] / ls, 'sqeuclidean')
        return var * np.exp(-0.5 * d)

    def __call__(self, X1, X2):
        K = self._rbf(X1, X2, self.ls_force, self.var_force, FORCE_INDICES)
        K += self._rbf(X1, X2, self.ls_terrain, self.var_terrain, TERRAIN_INDICES)
        K += self._rbf(X1, X2, self.ls_ai, self.var_ai, [AI_INDEX])
        K += self._rbf(X1, X2, self.ls_phase, self.var_phase, [PHASE_INDEX])
        K += self._rbf(X1, X2, self.ls_interact, self.var_interact,
                       list(range(X1.shape[1])))
        return K

    def pathway_kernels(self, X1, X2):
        return {
            "force_ratio": self._rbf(X1, X2, self.ls_force, self.var_force, FORCE_INDICES),
            "terrain": self._rbf(X1, X2, self.ls_terrain, self.var_terrain, TERRAIN_INDICES),
            "red_ai": self._rbf(X1, X2, self.ls_ai, self.var_ai, [AI_INDEX]),
            "phase": self._rbf(X1, X2, self.ls_phase, self.var_phase, [PHASE_INDEX]),
            "interaction": self._rbf(X1, X2, self.ls_interact, self.var_interact,
                                     list(range(X1.shape[1]))),
        }


class CausalGP:
    def __init__(self, kernel=None):
        self.kernel = kernel or CausalKernel()
        self.X_train = None
        self.y_train = None
        self.y_mean = 0.0

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float64)
        self.y_mean = np.mean(y)
        self.y_train = np.array(y, dtype=np.float64) - self.y_mean
        K = self.kernel(self.X_train, self.X_train)
        K += self.kernel.noise_var * np.eye(len(y)) + 1e-6 * np.eye(len(y))
        self.L = cho_factor(K)
        self.alpha = cho_solve(self.L, self.y_train)

    def predict(self, X_new):
        """E[y | do(d, u)] and Var[y | do(d, u)]"""
        X_new = np.atleast_2d(X_new)
        Ks = self.kernel(X_new, self.X_train)
        Kss = self.kernel(X_new, X_new)
        mu = Ks @ self.alpha + self.y_mean
        v = cho_solve(self.L, Ks.T)
        var = np.diag(Kss) - np.sum(Ks * v.T, axis=1)
        return mu, np.maximum(var, 1e-6)

    # ============================================================
    # do-calculus operations
    # ============================================================

    def _build_du_matrix(self, d_vec, u_samples):
        """Construct full (d,u) input matrix for a fixed d and multiple u."""
        n = len(u_samples)
        X = np.zeros((n, len(D_INDICES) + len(U_INDICES)))
        X[:, D_INDICES] = d_vec
        X[:, U_INDICES] = u_samples
        return X

    def do_d_marginal(self, d_vec, u_samples):
        """
        E[y | do(d)] = E_u[ E[y | do(d), do(u)] ]

        Interventional marginal effect of d, averaging over u distribution.
        Uses Monte Carlo integration over u_samples ~ P(u).
        """
        X = self._build_du_matrix(d_vec, u_samples)
        mu, var = self.predict(X)
        # Law of total expectation / total variance
        return float(np.mean(mu)), float(np.mean(var) + np.var(mu))

    def do_d_worst_u(self, d_vec, u_samples, beta=1.96):
        """
        Adversarial do-calculus:
        min_u E[y | do(d), do(u)]

        Uses lower confidence bound (LCB) for pessimistic estimate.
        Returns (worst_u_index, worst_lcb_value, all_lcb_values)
        """
        X = self._build_du_matrix(d_vec, u_samples)
        mu, var = self.predict(X)
        lcb = mu - beta * np.sqrt(var)
        worst_idx = np.argmin(lcb)
        return int(worst_idx), float(lcb[worst_idx]), lcb

    def causal_ate(self, d1_vec, d2_vec, u_samples):
        """
        Average Treatment Effect: ATE = E[y|do(d2)] - E[y|do(d1)]
        Decomposed per causal pathway.
        """
        mean1, _ = self.do_d_marginal(d1_vec, u_samples)
        mean2, _ = self.do_d_marginal(d2_vec, u_samples)

        # Per-pathway decomposition
        X1 = self._build_du_matrix(d1_vec, u_samples)
        X2 = self._build_du_matrix(d2_vec, u_samples)
        ate_pathway = {}
        pk1 = self.kernel.pathway_kernels(X1, self.X_train)
        pk2 = self.kernel.pathway_kernels(X2, self.X_train)
        for name in pk1:
            c1 = float(np.mean(pk1[name] @ self.alpha))
            c2 = float(np.mean(pk2[name] @ self.alpha))
            ate_pathway[name] = c2 - c1

        return mean2 - mean1, ate_pathway

    def causal_effect_decomposition(self, X_new):
        """Decompose E[y|do(d,u)] into per-pathway contributions."""
        X_new = np.atleast_2d(X_new)
        components = {}
        for name, Ks in self.kernel.pathway_kernels(X_new, self.X_train).items():
            components[name] = float((Ks @ self.alpha)[0])
        return components
