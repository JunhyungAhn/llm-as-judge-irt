"""Vendored CARE-SL (CARE-SVD, fully-Gaussian) aggregator from the official repo
SprocketLab/CARE (Zhao et al., arXiv:2603.00039), functions `learn_structure`, `get_weights`,
`caresl_aggregate` copied verbatim (only the unrelated module-level `snorkel` import in
pgm_tools.py is dropped). Plus a thin `care_aggregate(mat)` wrapper for our (N x J) score
matrices: mean-impute NaN, standardize, sign-align to pool mean, return NaN on solver failure.

apple-to-apple: input is the judge score matrix only (gold-free, continuous scores). gamma=1 is
fixed a-priori (NOT tuned on gold) to keep the comparison leakage-free.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import scipy as sp
import cvxpy as cp

warnings.filterwarnings("ignore")


# ───────────────────── verbatim from SprocketLab/CARE src/pgm_tools.py ─────────────────────
def learn_structure(sigma_O, gamma=3, solver=cp.SCS, verbose=True, **solver_kwargs):
    """Sparse + low-rank precision decomposition with optional solver diagnostics."""
    A = np.asarray(sigma_O)
    M = A.shape[0]
    lam = 4e-3 / np.sqrt(M)

    O = 0.5 * (A + A.T)
    O_root = np.real(sp.linalg.sqrtm(O))

    L_cvx = cp.Variable((M, M), PSD=True)
    S = cp.Variable((M, M), PSD=True)
    R = cp.Variable((M, M), PSD=True)

    objective = cp.Minimize(0.5 * cp.norm(R @ O_root, 'fro') ** 2 - cp.trace(R)
                            + lam * (gamma * cp.pnorm(S, 1) + cp.norm(L_cvx, "nuc")))
    problem = cp.Problem(objective, [R == S - L_cvx, L_cvx >> 0])
    try:
        if solver is None:
            problem.solve(verbose=verbose, **solver_kwargs)
        else:
            problem.solve(verbose=verbose, solver=solver, **solver_kwargs)
    except Exception:
        pass
    return S.value, L_cvx.value


def get_weights(L_est, penalize_dependency=True):
    eigenval, eigenvec = np.linalg.eig(L_est)
    idx = np.argsort(eigenval)[::-1]
    eigenval, eigenvec = eigenval[idx], eigenvec[:, idx]
    # take the leading factor (assumed = Q) as starting weights
    weights = eigenvec[:, 0] * np.sqrt(eigenval[0])
    if penalize_dependency:
        for i in range(1, len(eigenval)):
            if eigenval[i] < 0:
                continue
            spurious = eigenvec[:, i] * np.sqrt(eigenval[i])
            weights = weights - spurious
    return weights


def caresl_aggregate(df, gamma=1, solver=cp.SCS, verbose=False, corr_matrix=None,
                     weights=None, return_weights=False, penalize_dependency=True, **solver_kwargs):
    """CARESL aggregation with tunable sparsity weight ``gamma``."""
    if weights is None:
        if corr_matrix is None:
            corr_matrix = df.corr()
        S_est, L_est = learn_structure(corr_matrix, gamma=gamma, solver=solver,
                                       verbose=verbose, **solver_kwargs)
        weights = get_weights(L_est, penalize_dependency=penalize_dependency)
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != df.shape[1]:
        raise ValueError("CARESL weights length mismatch")
    denominator = np.sum(weights)
    if np.isclose(denominator, 0.0):
        raise ValueError("CARESL weights sum to zero")
    weighted_avg = sum(weights[i] * df.iloc[:, i].to_numpy() for i in range(len(weights))) / denominator
    if return_weights:
        return weighted_avg, weights
    return weighted_avg
# ───────────────────────────────────────────────────────────────────────────────────────────


def _std(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def _sanitize_correlation(corr):
    """Match official pgm_tools.sanitize_correlation (finite entries, unit diagonal)."""
    a = corr.to_numpy(copy=True)
    a[~np.isfinite(a)] = 0.0
    np.fill_diagonal(a, 1.0)
    return pd.DataFrame(a)


# official fully_gaussian_main.py passes max_iters=10000 to the SCS solver
SOLVER_KW = dict(max_iters=10000)


def care_aggregate(mat, gamma=1.0):
    """CARE-SVD on an (N x J) score matrix -> standardized quality vector (NaN on failure).

    Matches the official pipeline: sanitized correlation + SCS max_iters=10000. gamma fixed
    a-priori (gold-free). Use gamma=1 (library default) for the leakage-free comparison.
    """
    M = np.asarray(mat, float).copy()
    col = np.nanmean(M, axis=0)
    col = np.where(np.isnan(col), 0.0, col)
    nan_idx = np.where(np.isnan(M))
    M[nan_idx] = np.take(col, nan_idx[1])
    if np.allclose(np.nanstd(M, axis=0), 0):
        return np.full(M.shape[0], np.nan)
    try:
        corr = _sanitize_correlation(pd.DataFrame(M).corr())
        S, L = learn_structure(corr, gamma=gamma, solver=cp.SCS, verbose=False, **SOLVER_KW)
        w = np.asarray(get_weights(L), dtype=float)
        if not np.isfinite(w).all() or np.isclose(w.sum(), 0.0):
            return np.full(M.shape[0], np.nan)
        q = (M * w[None, :]).sum(1) / w.sum()
        q = _std(q.real if np.iscomplexobj(q) else q)
        from scipy.stats import spearmanr
        s = spearmanr(q, _std(np.nanmean(mat, axis=1))).statistic
        if s is not None and s < 0:
            q = -q
        return q
    except Exception:
        return np.full(M.shape[0], np.nan)
