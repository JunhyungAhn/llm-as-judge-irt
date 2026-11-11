# AGORA: Reliability-Aware Aggregation of LLM Judges on Small Evaluation Sets

[![arXiv](https://img.shields.io/badge/arXiv-2611.05556-b31b1b.svg)](https://arxiv.org/abs/2611.05556)

Reference implementation for the paper's main results. **AGORA** aggregates a matrix of
LLM-judge scores into a single quality estimate per item by fitting a graded-response
item-response-theory (IRT) model whose per-judge discrimination `α_j` carries a
regularized-horseshoe prior — so unreliable or low-information judges shrink toward zero and
the posterior-mean latent trait `θ̂` becomes a reliability-weighted consensus. No human labels
are used in the aggregation; gold is touched only when scoring the final correlation.

This repository holds the code only: the `agora/` package and a single entry point `main.py`.
The benchmark data and the collected judge scores are distributed as a release archive (see
Quick start).

## Quick start

Requires Python ≥ 3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                     # install pinned dependencies from uv.lock

# benchmark data + collected LLM-judge scores (unpacks into data/ and results/)
curl -LO https://github.com/JunhyungAhn/llm-as-judge-irt/releases/download/v1.0/agora-data-v1.0.zip
unzip agora-data-v1.0.zip

uv run python main.py       # reproduce the full main table (all methods x 3 datasets)
```

No API keys or network access are needed at run time — every benchmark reads the local files
from the archive, including the WMT22 MQM reference in `data/wmt22_ende_mqm_gold.json`.

### Running a single method or dataset

`main.py` is a [`fire`](https://github.com/google/python-fire) CLI. With no arguments it runs
all 8 methods on all 3 datasets and writes `results/agora/main_table.json`. Pass `--model` and/or
`--dataset` to run just one (this skips the other methods, so it is much faster):

```bash
uv run python main.py --model AGORA                 # AGORA only, all datasets
uv run python main.py --model AGORA --dataset SummEval
uv run python main.py DS WMT22                       # positional also works
```

- `--model`: `Mean` · `Median` · `Trim` · `DS` (Dawid–Skene) · `MACE` · `GLAD` · `CARE` · `AGORA`
  (case/hyphen-insensitive).
- `--dataset`: `SummEval` · `WMT22` · `Topical-Chat` (aliases: `se`, `wmt`, `tc`, …).

A filtered run prints its table and LaTeX row(s) but leaves the canonical
`results/agora/main_table.json` untouched; only the full no-argument run rewrites it.

## Main results (Table 1)

Dimension-averaged Spearman ρ vs. human references on the heterogeneous five-judge pool
(`front_low5`: 1 frontier + 4 open-weight judges), full sample / small sample (N = 50):

| Method       | Topical-Chat | WMT22       | SummEval    |
|--------------|:------------:|:-----------:|:-----------:|
| Mean         | 0.672 / 0.680 | 0.416 / 0.407 | 0.521 / 0.501 |
| Median       | 0.643 / 0.656 | 0.406 / 0.413 | 0.523 / 0.505 |
| Trimmed Mean | 0.674 / 0.684 | 0.418 / 0.400 | 0.538 / 0.518 |
| Dawid–Skene  | 0.695 / 0.683 | 0.423 / 0.399 | 0.555 / 0.504 |
| CARE         | 0.688 / 0.676 | 0.418 / 0.409 | 0.544 / 0.507 |
| GLAD         | 0.685 / 0.690 | 0.419 / 0.409 | 0.545 / 0.521 |
| MACE         | 0.690 / 0.692 | 0.418 / 0.408 | 0.533 / 0.522 |
| **AGORA**    | **0.705 / 0.698** | **0.426** / 0.412 | **0.556 / 0.527** |

The small sample (N = 50) refits every method on the same 40 random item subsets
(seeds 1000–1039), so the comparison is paired.

## Repository layout

```
agora/                 # the AGORA package
  model.py             #   fit_horseshoe_grm / agora_predict — the horseshoe-α GRM fit + EAP read-out
  grm.py               #   graded-response model primitives (log-probs, E-step, M-step, fit)
  baselines.py         #   Mean, Median, Trimmed mean, Dawid-Skene, MACE, GLAD, CARE
  care_svd.py          #   vendored CARE-SVD aggregator
  data.py              #   dataset loaders + the front_low5 judge pool
  metrics.py           #   Spearman rho, standardization, paired bootstrap
main.py                # fire CLI entry point — reproduces Table 1
data/, results/        # from agora-data-v1.0.zip, not tracked in git (judge checkpoints,
                       # USR data, SummEval cache, WMT22 MQM reference)
pyproject.toml, uv.lock
```

## Datasets

| Dataset       | Task              |     N | Dims | Gold                    |
|---------------|-------------------|------:|-----:|-------------------------|
| SummEval      | summarization     | 1,600 |    4 | 3 expert annotators     |
| Topical-Chat  | dialogue (USR)    |   360 |    6 | 3 annotators            |
| WMT22 en-de   | translation (MQM) | 19,725 |   1 | professional translators |

All three are evaluated on the same 14-judge pool; the paper's main table uses the
heterogeneous five-judge subset `front_low5`.

## Programmatic use

```python
from agora import agora_predict, load_frontlow5, rho

for dim, mat, gold, lo, K in load_frontlow5("SummEval"):
    pred = agora_predict(mat, lo, K)   # gold-free aggregation of the (items x judges) score matrix
    print(dim, rho(pred, gold))         # gold used only to score
```

## Citation

```bibtex
@article{ahn2026agora,
  title   = {{AGORA}: Reliability-Aware Aggregation of {LLM} Judges on Small Evaluation Sets},
  author  = {Ahn, Junhyung},
  journal = {arXiv preprint arXiv:2611.05556},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
