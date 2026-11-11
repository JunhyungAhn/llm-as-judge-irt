"""AGORA -- reliability-aware aggregation of LLM judges on small evaluation sets.

Reference implementation for the paper's main results (Table 1). The public surface:

    from agora import agora_predict, baseline_predictions, load_frontlow5, rho

    dims = load_frontlow5("SummEval")            # [(dim, mat, gold, lo, K), ...]
    for name, mat, gold, lo, K in dims:
        pred = agora_predict(mat, lo, K)          # gold-free
        score = rho(pred, gold)                    # gold only here

Reproduce the main table with ``python main.py`` (optionally ``--model`` / ``--dataset``).
"""
from .metrics import rho, pool_mean, paired_bootstrap, _std
from .model import agora_predict, fit_horseshoe_grm, eff_judges
from .baselines import baseline_predictions, baseline_predict, BASELINE_FNS
from .data import (
    load_summeval, load_tc, load_wmt, load_frontlow5,
    LOADERS, FRONT_LOW5, pool_idx, restrict,
)

__all__ = [
    "rho", "pool_mean", "paired_bootstrap", "_std",
    "agora_predict", "fit_horseshoe_grm", "eff_judges",
    "baseline_predictions", "baseline_predict", "BASELINE_FNS",
    "load_summeval", "load_tc", "load_wmt", "load_frontlow5",
    "LOADERS", "FRONT_LOW5", "pool_idx", "restrict",
]
