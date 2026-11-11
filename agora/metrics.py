"""Evaluation metrics for AGORA aggregation (gold used only here, in the final rho)."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _std(x: np.ndarray) -> np.ndarray:
    """Standardize to zero mean / unit variance (NaN-safe)."""
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def rho(pred: np.ndarray, gold: np.ndarray) -> float:
    """Spearman rho between a prediction and gold, over the jointly-observed items."""
    m = ~np.isnan(pred) & ~np.isnan(gold)
    return float(spearmanr(pred[m], gold[m]).statistic) if m.sum() >= 20 else float("nan")


def pool_mean(mat: np.ndarray) -> np.ndarray:
    """Per-item mean over judges, standardized."""
    return _std(np.nanmean(mat, axis=1))


def paired_bootstrap(pred_a: np.ndarray, pred_b: np.ndarray, gold: np.ndarray,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """Item-level paired bootstrap of rho(a) - rho(b). Returns mean diff, 95% CI, sig."""
    rng = np.random.default_rng(seed)
    N = len(gold)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        diffs[b] = rho(pred_a[idx], gold[idx]) - rho(pred_b[idx], gold[idx])
    lo_c, hi_c = np.nanpercentile(diffs, [2.5, 97.5])
    return {"mean_diff": round(float(np.nanmean(diffs)), 4),
            "ci95": [round(float(lo_c), 4), round(float(hi_c), 4)],
            "sig": bool(lo_c > 0 or hi_c < 0)}
