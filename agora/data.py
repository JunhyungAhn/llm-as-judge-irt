"""Dataset loaders and the heterogeneous five-judge pool (front_low5) for AGORA.

Each loader returns a list of ``(dim_name, mat, gold, lo, K)`` tuples, where ``mat`` is the
(N items x J judges) score matrix, ``gold`` the human reference vector, ``lo`` the minimum raw
score, and ``K`` the number of ordinal categories. Gold is returned only so the caller can score
the final aggregation -- no aggregation step reads it.

Inputs come from the release archive ``agora-data-v1.0.zip``, unpacked at the repository root:
``data/`` holds the benchmark texts and human references, and ``results/*/checkpoints/`` holds
the collected LLM-judge scores (one JSON list of ``{"sample_id", "score"}`` per judge and dimension).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "results"

# ----- SummEval: 4 dimensions, N=1600, scores 1-5 -----

SE_DIMS = ["coherence", "consistency", "fluency", "relevance"]
SE_CKPT = CKPT / "summeval_with_source" / "checkpoints"
SE_JUDGES = [
    "DeepSeek-V3", "di-Qwen3-Next-80B-A3B-Instruct", "di-Llama-4-Scout-17B-16E-Instruct",
    "DeepSeek-R1-0528", "di-gpt-oss-120b", "Mixtral-8x22B", "Phi-4", "Qwen2.5-72B-Instruct",
    "gpt-5", "gemini-2.5-pro", "global.anthropic.claude-sonnet-4-6",
    "di-gemma-4-31B-it", "GLM-4.7-Flash", "Nemotron-120B",
]


def _se_scores(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return {r["sample_id"]: int(r["score"])
            for r in json.loads(path.read_text()) if r.get("score", -1) > 0}


def _se_gold(dim: str) -> dict[str, float]:
    """Mean of the three expert annotations per summary."""
    lines = (DATA / "summeval_raw.jsonl").read_text().strip().split("\n")
    per: dict[str, list[int]] = defaultdict(list)
    for idx, line in enumerate(lines):
        for ann in json.loads(line).get("expert_annotations", []):
            score = ann.get(dim)
            if score is not None:
                per[f"summeval_{idx:04d}"].append(int(score))
    return {sid: float(np.mean(v)) for sid, v in per.items()}


def load_summeval():
    """SummEval: 4 dimensions, N=1600, K=5 (raw 1-5)."""
    out = []
    for dim in SE_DIMS:
        gd = _se_gold(dim)
        sids = sorted(gd)
        idx = {s: i for i, s in enumerate(sids)}
        mat = np.full((len(sids), len(SE_JUDGES)), np.nan)
        for j, model in enumerate(SE_JUDGES):
            for sid, v in _se_scores(SE_CKPT / f"{dim}__{model}.json").items():
                if sid in idx:
                    mat[idx[sid], j] = float(v)
        out.append((dim, mat, np.array([gd[s] for s in sids], float), 1, 5))
    return out


# ----- Topical-Chat USR: 6 dimensions, N=360, mixed scales -----

TC_USR = DATA / "topical_chat" / "tc_usr_data.json"
TC_CKPT = CKPT / "topical_chat" / "checkpoints"
TC_DIMS = [
    # (human field, slug, lo, hi, K)
    ("Understandable",    "understandable",    0, 1, 2),
    ("Natural",           "natural",           1, 3, 3),
    ("Maintains Context", "maintains_context", 1, 3, 3),
    ("Engaging",          "engaging",          1, 3, 3),
    ("Uses Knowledge",    "uses_knowledge",    0, 1, 2),
    ("Overall",           "overall",           1, 5, 5),
]
TC_JUDGES = [
    "DeepSeek-V3", "di-Qwen3-Next-80B-A3B-Instruct", "di-Llama-4-Scout-17B-16E-Instruct",
    "DeepSeek-R1-0528", "di-gpt-oss-120b", "Mixtral-8x22B", "Phi-4", "Qwen2.5-72B-Instruct",
    "di-gemma-4-31B-it", "GLM-4.7-Flash", "Nemotron-120B",
    "gpt-5", "gemini-2.5-pro", "claude-sonnet-4-6",
]


def load_tc():
    """Topical-Chat USR: 6 dimensions, N=360, mixed K (raw 0-1 / 1-3 / 1-5)."""
    raw = json.loads(TC_USR.read_text())
    sids = [f"tc_{c:03d}_{r}" for c, ctx in enumerate(raw) for r in range(len(ctx["responses"]))]
    idx = {s: i for i, s in enumerate(sids)}
    out = []
    for field, slug, lo, hi, K in TC_DIMS:
        gd = {f"tc_{c:03d}_{r}": float(np.mean(resp[field]))
              for c, ctx in enumerate(raw) for r, resp in enumerate(ctx["responses"])}
        mat = np.full((len(sids), len(TC_JUDGES)), np.nan)
        for j, model in enumerate(TC_JUDGES):
            path = TC_CKPT / f"{slug}__{model}.json"
            if not path.exists():
                continue
            for r in json.loads(path.read_text()):
                sc = r.get("score")
                if isinstance(sc, int) and lo <= sc <= hi and r["sample_id"] in idx:
                    mat[idx[r["sample_id"]], j] = float(sc)
        out.append((slug, mat, np.array([gd[s] for s in sids], float), lo, K))
    return out


# ----- WMT22 en-de: 1 dimension (MQM), N=19,725, scores 1-5 -----

WMT_JUDGES = {
    "gpt-5":                             "wmt_frontier_openai",
    "gemini-2.5-pro":                    "wmt_frontier_api",
    "claude-sonnet-4-6":                 "wmt_frontier_api",
    "DeepSeek-V3":                       "wmt_cheap_deepinfra",
    "di-Qwen3-Next-80B-A3B-Instruct":    "wmt_cheap_deepinfra",
    "di-Llama-4-Scout-17B-16E-Instruct": "wmt_cheap_deepinfra",
    "DeepSeek-R1-0528":                  "wmt_cheap_deepinfra",
    "di-gpt-oss-120b":                   "wmt_cheap_deepinfra",
    "Mixtral-8x22B":                     "wmt_cheap_deepinfra",
    "Phi-4":                             "wmt_cheap_deepinfra",
    "Qwen2.5-72B-Instruct":              "wmt_cheap_deepinfra",
    "di-gemma-4-31B-it":                 "wmt_cheap_deepinfra",
    "GLM-4.7-Flash":                     "wmt_cheap_deepinfra",
    "Nemotron-120B":                     "wmt_cheap_deepinfra",
}
# MQM references in the segment order the judges scored, extracted from the HuggingFace dataset
# `RicardoRei/wmt-mqm-human-evaluation` (year == 2022, lp == "en-de", non-null score).
WMT_GOLD = DATA / "wmt22_ende_mqm_gold.json"


def load_wmt():
    """WMT22 en-de: 1 dimension (MQM), N=19,725, K=5 (raw 1-5)."""
    gold = np.array(json.loads(WMT_GOLD.read_text()), dtype=float)
    idx = {f"wmt22_ende_{i:06d}": i for i in range(len(gold))}
    mat = np.full((len(gold), len(WMT_JUDGES)), np.nan)
    for j, (model, folder) in enumerate(WMT_JUDGES.items()):
        path = CKPT / folder / "checkpoints" / f"{model}.json"
        if not path.exists():
            continue
        for r in json.loads(path.read_text()):
            sc = r.get("score")
            if isinstance(sc, int) and 1 <= sc <= 5 and r["sample_id"] in idx:
                mat[idx[r["sample_id"]], j] = float(sc)
    return [("mqm", mat, gold, 1, 5)]


LOADERS = {"SummEval": load_summeval, "Topical-Chat": load_tc, "WMT22": load_wmt}
JUDGE_NAMES = {"SummEval": SE_JUDGES, "Topical-Chat": TC_JUDGES, "WMT22": list(WMT_JUDGES)}


# ----- heterogeneous five-judge pool (front_low5): 1 frontier + 4 open-weight, fixed by name -----

FRONT_LOW5 = ["Llama-4-Scout", "DeepSeek-R1", "claude-sonnet-4-6", "GLM-4.7-Flash", "Nemotron-120B"]


def pool_idx(names: list[str]) -> list[int]:
    """Column indices of the five front_low5 judges within a dataset's judge ordering."""
    out = []
    for sub in FRONT_LOW5:
        m = [i for i, n in enumerate(names) if sub in n]
        if not m:
            raise ValueError(f"model {sub} not found in {names}")
        out.append(m[0])
    return out


def restrict(dims, cols):
    """Restrict every dimension's score matrix to the given judge columns."""
    return [(d, mat[:, cols], gold, lo, K) for d, mat, gold, lo, K in dims]


def load_frontlow5(name: str):
    """Load a dataset already restricted to the heterogeneous five-judge pool."""
    return restrict(LOADERS[name](), pool_idx(JUDGE_NAMES[name]))
