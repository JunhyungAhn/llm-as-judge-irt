"""AGORA: adaptive-sparsity graded-response IRT aggregator.

The judge score matrix is fit with a graded-response model whose per-judge discrimination
``alpha_j`` carries a regularized-horseshoe prior (Piironen-Vehtari 2017), estimated by MAP-EM with
the Makalic-Schmidt (2016) inverse-gamma augmentation of the half-Cauchy scales. The data set the
concentration of ``alpha`` per dimension: low-information or corrupted judges shrink toward zero,
so the posterior-mean (EAP) latent trait ``theta`` is a reliability-weighted consensus. Gold is
never used in the fit. The graded-response primitives live in ``agora.grm``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as sp_minimize

from .grm import grm_log_probs, e_step, fit_irt
from .metrics import _std, rho, pool_mean

C_SLAB = 5.0          # regularized-horseshoe slab scale (caps large alpha); fixed a-priori
N_ITER = 12


def _m_step_pen(theta, scores, lo, K, s2):
    """Penalized GRM M-step: Gaussian penalty 0.5*alpha^2/s2 (horseshoe conditional)."""
    valid = ~np.isnan(scores)
    if valid.sum() < 15:
        return 0.5, np.linspace(-1.5, 1.5, K - 1)
    tv = theta[valid]; sv = scores[valid].astype(int) - lo
    center = (K - 1) / 2.0
    inv = 1.0 / max(s2, 1e-6)

    def nll(p):
        a = p[0]; b = np.cumsum(np.exp(p[1:])) - center
        ll = -grm_log_probs(tv, a, b, K)[sv, np.arange(len(sv))].sum()
        return ll + 0.5 * a * a * inv
    x0 = [1.0] + [0.0] * (K - 1)
    bnds = [(0.01, 20.0)] + [(None, None)] * (K - 1)
    try:
        r = sp_minimize(nll, x0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 80})
        return float(r.x[0]), np.cumsum(np.exp(r.x[1:])) - center
    except Exception:
        return 0.5, np.linspace(-1.5, 1.5, K - 1)


def fit_horseshoe_grm(mat, lo, K, n_iter=N_ITER, c_slab=C_SLAB):
    """Adaptive-sparsity GRM. Returns (theta_std, alphas, tau)."""
    J = mat.shape[1]
    alphas, betas, theta = fit_irt(mat, lo, K, n_iter=6)   # warm start
    theta = _std(theta)
    lam2 = np.ones(J); nu = np.ones(J); tau2 = 1.0; xi = 1.0
    c2 = c_slab ** 2
    for _ in range(n_iter):
        # ---- update horseshoe scales (Makalic-Schmidt conditional modes) ----
        a2 = alphas ** 2
        lam2 = (1.0 / nu + a2 / (2.0 * tau2)) / 2.0            # InvGamma(1,b) mode=b/2
        nu = (1.0 + 1.0 / np.maximum(lam2, 1e-8)) / 2.0
        rate_tau = 1.0 / xi + np.sum(a2 / (2.0 * np.maximum(lam2, 1e-8)))
        tau2 = rate_tau / ((J + 1) / 2.0 + 1.0)                # InvGamma(shape,b) mode
        xi = (1.0 + 1.0 / max(tau2, 1e-8)) / 2.0
        lam2_t = c2 * lam2 / (c2 + tau2 * lam2)                # regularized (slab-capped)
        s2 = tau2 * lam2_t                                     # effective prior var on alpha_j
        # ---- penalized M-step ----
        for j in range(J):
            alphas[j], betas[j] = _m_step_pen(theta, mat[:, j], lo, K, s2[j])
        # ---- E-step ----
        theta = e_step(mat, alphas, betas, np.ones(J), lo, K)
        theta = _std(theta)
    theta = _std(theta)
    if rho(theta, pool_mean(mat)) < 0:
        theta = -theta
    return theta, alphas, float(np.sqrt(tau2))


def agora_predict(mat, lo, K) -> np.ndarray:
    """AGORA quality estimate for one score matrix: the standardized latent trait theta."""
    return fit_horseshoe_grm(mat, lo, K)[0]


def eff_judges(alphas) -> float:
    """Participation ratio (sum|a|)^2 / sum a^2 = effective number of active judges."""
    a = np.abs(alphas)
    return float((a.sum() ** 2) / (np.sum(a ** 2) + 1e-12))
