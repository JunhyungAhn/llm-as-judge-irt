"""Reference aggregators compared against AGORA in the main table.

All operate on the raw (N x J) judge score matrix only (gold-free) and return a standardized
quality vector.

  Mean          : per-item mean over judges.
  Median        : per-item median.
  Trimmed Mean  : 20% symmetric trimmed mean.
  Dawid-Skene   : crowdkit categorical DS, posterior-expected label.
  MACE          : crowdkit MACE.
  GLAD          : crowdkit GLAD.
  CARE          : CARE-SVD (fully-Gaussian), gamma=1 fixed a-priori.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from crowdkit.aggregation import DawidSkene, MACE, GLAD

from .metrics import _std, pool_mean
from .care_svd import care_aggregate

CK = 10   # crowdkit EM iterations


def _ck(agg, mat):
    """Run a crowdkit categorical aggregator and return the posterior-expected label per item."""
    N = mat.shape[0]
    rows = [(int(i), int(j), int(round(mat[i, j]))) for j in range(mat.shape[1])
            for i in np.where(~np.isnan(mat[:, j]))[0]]
    try:
        pr = agg.fit_predict_proba(pd.DataFrame(rows, columns=["task", "worker", "label"]))
    except Exception:
        return np.full(N, np.nan)
    lab = np.array([float(c) for c in pr.columns]); e = (pr.values * lab[None, :]).sum(1)
    o = np.full(N, np.nan)
    for t, val in zip(pr.index, e):
        o[int(t)] = val
    return _std(o)


def _trim(mat, frac=0.2):
    out = np.full(mat.shape[0], np.nan)
    for i in range(mat.shape[0]):
        row = np.sort(mat[i, ~np.isnan(mat[i])])
        if len(row) == 0:
            continue
        t = int(np.floor(frac * len(row)))
        out[i] = np.mean(row[t:len(row) - t]) if len(row) - 2 * t > 0 else np.mean(row)
    return _std(out)


# name -> single-method predictor (each takes the score matrix, returns a standardized vector).
# Kept as thunks so selecting one method never runs the others (crowdkit/CARE are the slow ones).
BASELINE_FNS = {
    "Mean": lambda mat: pool_mean(mat),
    "Median": lambda mat: _std(np.nanmedian(mat, axis=1)),
    "Trimmed Mean": lambda mat: _trim(mat),
    "Dawid-Skene": lambda mat: _ck(DawidSkene(n_iter=CK), mat),
    "MACE": lambda mat: _ck(MACE(n_iter=CK, n_restarts=1), mat),
    "GLAD": lambda mat: _ck(GLAD(n_iter=CK, m_step_max_iter=5), mat),
    "CARE": lambda mat: care_aggregate(mat),
}


def baseline_predict(name: str, mat) -> np.ndarray:
    """Prediction for one named baseline (computes only that method)."""
    return BASELINE_FNS[name](mat)


def baseline_predictions(mat) -> dict[str, np.ndarray]:
    """The seven non-AGORA rows of the main table for one score matrix."""
    return {name: fn(mat) for name, fn in BASELINE_FNS.items()}
