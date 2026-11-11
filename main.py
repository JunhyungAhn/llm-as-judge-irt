"""Reproduce the main results table (Table 1) of the AGORA paper.

Single source of truth for ``tab:main``: dimension-averaged Spearman rho against human references
on the heterogeneous five-judge pool (front_low5), for every method at the full sample and the
shared small sample (N=50). Deterministic -- the N=50 column re-fits each method on the SAME 40
random item subsets (seed 1000+s), so all methods are paired. Gold is used only in the final rho.

Usage (fire CLI):

    python main.py                         # all 8 methods x 3 datasets, print + LaTeX + JSON
    python main.py --model AGORA           # only AGORA (skips the slow crowdkit/CARE baselines)
    python main.py --model DS --dataset WMT22
    python main.py AGORA SummEval          # positional also works

``--model`` selects one aggregator (Mean / Median / Trim / DS / MACE / GLAD / CARE / AGORA;
aliases accepted). ``--dataset`` selects one benchmark (SummEval / WMT22 / Topical-Chat). With no
arguments every method and dataset is run and the canonical results/agora/main_table.json is
written; a filtered run prints only, leaving that file untouched.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import fire
import numpy as np

from agora import agora_predict, baseline_predict, load_frontlow5, rho

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results" / "agora"

# tab:main row order and column (dataset) order
METHODS = ["Mean", "Median", "Trimmed Mean", "Dawid-Skene", "CARE", "GLAD", "MACE", "AGORA"]
DATASETS = ["Topical-Chat", "WMT22", "SummEval"]
FULL_N = {"Topical-Chat": 360, "WMT22": 19725, "SummEval": 1600}
SMALL_N = 50
SEEDS = 40           # paired subsamples shared by every method (seed base 1000)
SEED_BASE = 1000

# CLI aliases -> canonical names (case/space/hyphen-insensitive)
_METHOD_ALIASES = {
    "mean": "Mean", "median": "Median",
    "trim": "Trimmed Mean", "trimmed": "Trimmed Mean", "trimmedmean": "Trimmed Mean",
    "ds": "Dawid-Skene", "dawidskene": "Dawid-Skene",
    "mace": "MACE", "glad": "GLAD", "care": "CARE",
    "agora": "AGORA",
}
_DATASET_ALIASES = {
    "summeval": "SummEval", "se": "SummEval",
    "wmt": "WMT22", "wmt22": "WMT22",
    "topicalchat": "Topical-Chat", "tc": "Topical-Chat", "topical": "Topical-Chat",
}


def _resolve(value: str, aliases: dict[str, str], kind: str) -> str:
    key = str(value).lower().replace(" ", "").replace("-", "").replace("_", "")
    if key not in aliases:
        raise ValueError(f"unknown {kind} {value!r}; choose from {sorted(set(aliases.values()))}")
    return aliases[key]


def predict_one(method: str, mat, lo, K) -> np.ndarray:
    """Prediction for a single aggregator (computes only that method)."""
    return agora_predict(mat, lo, K) if method == "AGORA" else baseline_predict(method, mat)


def _dim_avg_rho(dims, methods, idx=None) -> dict[str, float]:
    """Dimension-averaged rho per method; idx optionally selects an item subset."""
    per = {m: [] for m in methods}
    for _, mat0, gold, lo, K in dims:
        mat = mat0 if idx is None else mat0[idx]
        g = gold if idx is None else gold[idx]
        for m in methods:
            per[m].append(rho(predict_one(m, mat, lo, K), g))
    return {m: float(np.nanmean(per[m])) for m in methods}


def evaluate(name: str, methods) -> dict[str, dict[str, float]]:
    """Full-sample and small-sample (N=50, mean over shared subsets) rho for one dataset."""
    dims = load_frontlow5(name)
    Nfull = dims[0][1].shape[0]

    full = _dim_avg_rho(dims, methods)

    small_runs = {m: [] for m in methods}
    for s in range(SEEDS):
        rng = np.random.default_rng(SEED_BASE + s)
        idx = rng.choice(Nfull, SMALL_N, replace=False)
        vals = _dim_avg_rho(dims, methods, idx=idx)
        for m in methods:
            small_runs[m].append(vals[m])
    small = {m: float(np.nanmean(small_runs[m])) for m in methods}

    return {"full": {m: round(full[m], 4) for m in methods},
            "small": {m: round(small[m], 4) for m in methods}}


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def print_table(results: dict, methods, datasets) -> None:
    """Console rendering: full / small columns per dataset."""
    header = f"{'Method':<14}" + "".join(f"{d + ' f/s':>22}" for d in datasets)
    print("\n" + header)
    print("-" * len(header))
    for m in methods:
        cells = "".join(f"{_fmt(results[d]['full'][m]) + ' / ' + _fmt(results[d]['small'][m]):>22}"
                        for d in datasets)
        print(f"{m:<14}{cells}")


def latex_body(results: dict, methods, datasets) -> str:
    """LaTeX rows matching tab:main (bold = best point estimate per column when >1 method)."""
    best = {(d, col): max(results[d][col][m] for m in methods)
            for d in datasets for col in ("full", "small")}
    label = {"Trimmed Mean": "Trimmed Mean", "Dawid-Skene": "Dawid--Skene", "AGORA": "\\MODEL{}"}
    lines = []
    for m in methods:
        cells = []
        for d in datasets:
            for col in ("full", "small"):
                v = results[d][col][m]
                s = _fmt(v)
                if len(methods) > 1 and abs(v - best[(d, col)]) < 5e-4:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
        lines.append(f"  {label.get(m, m):<20} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines)


def main(model: str | None = None, dataset: str | None = None) -> None:
    """Regenerate the AGORA main table, optionally for a single model and/or dataset.

    Args:
        model:   one aggregator to run (Mean/Median/Trim/DS/MACE/GLAD/CARE/AGORA). Default: all.
        dataset: one benchmark to run (SummEval/WMT22/Topical-Chat). Default: all.
    """
    sel_methods = {_resolve(model, _METHOD_ALIASES, "model")} if model else set(METHODS)
    methods = [m for m in METHODS if m in sel_methods]        # keep canonical order
    sel_datasets = {_resolve(dataset, _DATASET_ALIASES, "dataset")} if dataset else set(DATASETS)
    datasets = [d for d in DATASETS if d in sel_datasets]
    full_run = model is None and dataset is None

    results = {}
    for name in datasets:
        print(f"[main] evaluating {name} (front_low5, full N={FULL_N[name]} / small N={SMALL_N}) "
              f"for {', '.join(methods)} ...")
        results[name] = evaluate(name, methods)

    print_table(results, methods, datasets)
    print("\n% ---- LaTeX body for tab:main ----")
    print(latex_body(results, methods, datasets))

    if full_run:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        payload = {"pool": "front_low5", "seeds": SEEDS, "small_n": SMALL_N,
                   "methods": METHODS, "datasets": DATASETS, "results": results}
        (OUTDIR / "main_table.json").write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {OUTDIR / 'main_table.json'}")
    else:
        print("\n(filtered run: canonical results/agora/main_table.json left untouched)")


if __name__ == "__main__":
    fire.Fire(main)
