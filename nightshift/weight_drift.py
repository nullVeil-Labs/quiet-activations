"""Idea 2, exploratory: relative weight change from base to instruct (protocols/idea2_weight_drift_exploratory.md)."""
import glob
import re
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open

from .common import ROOT, save_json

PAIRS = {"qwen3-1.7b": ("qwen3-1.7b-base", "qwen3-1.7b"), "smollm2-1.7b": ("SmolLM2-1b-base", "smollm2-1.7b-instruct")}


def tensors(folder):
    out = {}
    for f in sorted(glob.glob(str(ROOT / "models" / folder / "*.safetensors"))):
        with safe_open(f, framework="pt") as s:
            for k in s.keys():
                out[k] = (f, k)
    return out


def load(ref):
    with safe_open(ref[0], framework="pt") as s:
        return s.get_tensor(ref[1]).float().cuda()


def group(name):
    if re.search(r"mlp\.(gate|up|down)_proj", name):
        return "mlp"
    if re.search(r"self_attn\.(q|k|v|o)_proj", name):
        return "attention"
    if "norm" in name:
        return "norm"
    if "embed" in name or "lm_head" in name:
        return "embed"
    return "other"


def run(out_dir):
    res = {}
    for pair, (b, i) in PAIRS.items():
        tb, ti = tensors(b), tensors(i)
        shared = sorted(set(tb) & set(ti))
        by_group, down = defaultdict(list), {}
        for k in shared:
            wb, wi = load(tb[k]), load(ti[k])
            if wb.shape != wi.shape:
                continue
            r = ((wi - wb).norm() / wb.norm().clamp(min=1e-12)).item()
            by_group[group(k)].append(r)
            m = re.search(r"layers\.(\d+)\.mlp\.down_proj", k)
            if m:
                down[int(m.group(1))] = r
        res[pair] = {"n_shared_tensors": len(shared), "only_in_base": sorted(set(tb) - set(ti)),
                     "only_in_instruct": sorted(set(ti) - set(tb)),
                     "groups": {g: {"n": len(v), "median_r": float(np.median(v)), "max_r": float(np.max(v))}
                                for g, v in by_group.items()},
                     "down_proj_r_per_layer": [down[l] for l in sorted(down)]}
        print(pair, {g: round(v["median_r"], 5) for g, v in res[pair]["groups"].items()})
    save_json(out_dir / "idea2_weight_drift.json", res)
    return res
