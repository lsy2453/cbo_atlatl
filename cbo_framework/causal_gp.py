"""
Gaussian-process surrogate for single-level causal Bayesian optimization.

The GP learns the marginal interventional response

    f(x) = E_epsilon[y | do(x)]

where epsilon is the exogenous scenario-generation seed. The causal graph is
used for intervention analysis and path-oriented interpretation, while the
surrogate itself uses a standard RBF kernel.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


class RBFKernel:
    """Standard anisotropic RBF kernel over the encoded scenario vector."""

    def __init__(self, lengthscale=None, variance=1.0, noise_var=1e-4):
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_var = noise_var

    def __call__(self, X1, X2):
        X1 = np.atleast_2d(X1).astype(np.float64)
        X2 = np.atleast_2d(X2).astype(np.float64)
        if self.lengthscale is None:
            ls = np.ones(X1.shape[1], dtype=np.float64)
        else:
            ls = np.asarray(self.lengthscale, dtype=np.float64)
        d = cdist(X1 / ls, X2 / ls, "sqeuclidean")
        return self.variance * np.exp(-0.5 * d)


class CausalGP:
    """GP surrogate plus Monte Carlo intervention utilities."""

    def __init__(self, kernel=None):
        self.kernel = kernel or RBFKernel()
        self.X_train = None
        self.y_train = None
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.L = None
        self.alpha = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.y_mean = float(np.mean(y))
        self.y_scale = float(np.std(y))
        if self.y_scale < 1e-8:
            self.y_scale = 1.0
        self.y_train = (y - self.y_mean) / self.y_scale
        K = self.kernel(self.X_train, self.X_train)
        K += (self.kernel.noise_var + 1e-6) * np.eye(len(y))
        self.L = cho_factor(K, lower=True, check_finite=False)
        self.alpha = cho_solve(self.L, self.y_train, check_finite=False)

    def predict(self, X_new):
        """Return posterior mean and variance for E[y | do(x)]."""
        if self.X_train is None:
            raise RuntimeError("CausalGP must be fit before prediction.")
        X_new = np.atleast_2d(X_new).astype(np.float64)
        Ks = self.kernel(X_new, self.X_train)
        Kss = self.kernel(X_new, X_new)
        mu_std = Ks @ self.alpha
        v = cho_solve(self.L, Ks.T, check_finite=False)
        var_std = np.diag(Kss) - np.sum(Ks * v.T, axis=1)
        mu = mu_std * self.y_scale + self.y_mean
        var = np.maximum(var_std, 1e-8) * (self.y_scale ** 2)
        return mu, var

    def do_marginal(self, fixed_index, fixed_value, reference_samples):
        """
        Estimate E[y | do(x_i = a)] by Monte Carlo over reference samples.
        """
        X = np.array(reference_samples, dtype=np.float64, copy=True)
        X[:, fixed_index] = fixed_value
        mu, var = self.predict(X)
        return float(np.mean(mu)), float(np.mean(var) + np.var(mu))

    def causal_ate(self, index, value_from, value_to, reference_samples):
        """
        Average treatment effect of changing one coordinate:

            E[y | do(x_i=value_to)] - E[y | do(x_i=value_from)]
        """
        mean_from, _ = self.do_marginal(index, value_from, reference_samples)
        mean_to, _ = self.do_marginal(index, value_to, reference_samples)
        return float(mean_to - mean_from)

    def local_effect(self, x_vec, index, value_to):
        """Predicted local intervention effect at a specific scenario x."""
        x_from = np.array(x_vec, dtype=np.float64, copy=True)
        x_to = np.array(x_vec, dtype=np.float64, copy=True)
        x_to[index] = value_to
        mu_from, _ = self.predict(x_from.reshape(1, -1))
        mu_to, _ = self.predict(x_to.reshape(1, -1))
        return float(mu_to[0] - mu_from[0])


# Backward-compatible name for older imports.
CausalKernel = RBFKernel
