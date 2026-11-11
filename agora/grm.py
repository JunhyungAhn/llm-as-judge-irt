"""Graded-response model (GRM) primitives with an explicit number of categories K.

Scores are mapped to category indices by ``score - lo``. The latent trait is integrated on a
fixed grid, and the E-step returns the posterior mean (EAP).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as sp_minimize

THETA_GRID = np.linspace(-4.5, 4.5, 150)


def grm_log_probs(theta, alpha, beta, K: int):
    """K-categorical GRM log-prob: returns shape (K, T)."""
    def ls(x):
        x = np.clip(x, -30, 30)
        return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))
    lp = []
    for k in range(K):
        if k == 0:
            lp.append(ls(-alpha * (theta - beta[0])))
        elif k == K - 1:
            lp.append(ls(alpha * (theta - beta[K - 2])))
        else:
            l1 = ls(alpha * (theta - beta[k - 1]))
            l2 = ls(alpha * (theta - beta[k]))
            diff = np.expm1(np.clip(l2 - l1, -30, 0))
            lp.append(l1 + np.log(np.maximum(-diff, 1e-10)))
    return np.stack(lp)


def e_step(mat, alphas, betas, weights, lo: int, K: int):
    """Posterior-mean (EAP) latent trait per item over THETA_GRID."""
    n_s = mat.shape[0]
    lp = -0.5 * THETA_GRID[:, None] ** 2 * np.ones((len(THETA_GRID), n_s))
    for j in range(mat.shape[1]):
        w = float(weights[j])
        if w <= 0:
            continue
        valid = ~np.isnan(mat[:, j])
        if not valid.any():
            continue
        sc = mat[valid, j].astype(int) - lo
        g = grm_log_probs(THETA_GRID, max(alphas[j] * w, 0.01), betas[j], K)
        lp[:, np.where(valid)[0]] += g[sc, :].T
    lp -= lp.max(axis=0, keepdims=True)
    p = np.exp(lp); p /= p.sum(axis=0, keepdims=True)
    return (THETA_GRID[:, None] * p).sum(axis=0)


def m_step(theta, scores, lo: int, K: int):
    """Optimise alpha_j and the K-1 thresholds beta_j by L-BFGS-B."""
    valid = ~np.isnan(scores)
    if valid.sum() < 15:
        return 1.0, np.linspace(-1.5, 1.5, K - 1)
    tv = theta[valid]
    sv = scores[valid].astype(int) - lo
    center = (K - 1) / 2.0

    def nll(p):
        a = p[0]
        b = np.cumsum(np.exp(p[1:])) - center
        return -grm_log_probs(tv, a, b, K)[sv, np.arange(len(sv))].sum()

    x0 = [1.0] + [np.log(1.0)] * (K - 1)
    bounds = [(0.01, 20.0)] + [(None, None)] * (K - 1)
    try:
        r = sp_minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 100, "ftol": 1e-5})
        return float(r.x[0]), np.cumsum(np.exp(r.x[1:])) - center
    except Exception:
        return 1.0, np.linspace(-1.5, 1.5, K - 1)


def fit_irt(mat, lo: int, K: int, n_iter: int = 8,
            theta_init=None, weights=None):
    """Unpenalized GRM fit by alternating M- and E-steps. Returns (alphas, betas, thetas)."""
    J = mat.shape[1]
    if weights is None:
        weights = np.ones(J)
    rm = np.nanmean(mat, axis=1)
    rm = np.where(np.isnan(rm), (lo + K - 1) / 2.0 + lo, rm)
    thetas = ((rm - rm.mean()) / (rm.std() + 1e-8)
              if theta_init is None else theta_init.copy())
    thetas = (thetas - thetas.mean()) / (thetas.std() + 1e-8)
    alphas = np.ones(J)
    betas = [np.linspace(-1.5, 1.5, K - 1)] * J
    for _ in range(n_iter):
        for j in range(J):
            alphas[j], betas[j] = m_step(thetas, mat[:, j], lo, K)
        thetas = e_step(mat, alphas, betas, weights, lo, K)
        thetas = (thetas - thetas.mean()) / (thetas.std() + 1e-8)
    return alphas, betas, thetas
