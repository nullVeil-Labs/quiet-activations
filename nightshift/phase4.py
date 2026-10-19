"""Idea 2, Phase 4 (protocols/idea2_phase4.md): base vs instruct (A), and what an output norm would absorb (B)."""
import random

import numpy as np
import torch
import torch.nn.functional as F

from . import phase2 as p2
from .common import save_json
from .layer11 import NoOutlierHook

A_SLICE, B_SLICE = slice(250, 350), slice(350, 360)
A_MODELS = ["qwen3-1.7b-base", "qwen3-1.7b"]
B_MODELS = ["qwen3-1.7b-base", "qwen3-1.7b", "gemma-3-4b-it", "gemma-4"]
GUARD = 0.95


def texts():
    calib, a_set, b_set = [], [], []
    for book in p2.BOOKS:
        cs = p2.chunks(book)
        random.Random(0).shuffle(cs)  # identical shuffle to phases 2-3
        calib += cs[:p2.N_CALIB]
        a_set += cs[A_SLICE]
        b_set += cs[B_SLICE]
    return calib, a_set, b_set


class GuardedHook(NoOutlierHook):
    """Outlier-excluded quiet set; layers in `skip` are never ablated."""

    def __init__(self, downs):
        self.skip = set()
        super().__init__(downs)

    def make(self, li):
        inner = super().make(li)

        def hook(module, args, output):
            if li in self.skip and self.mode not in (None, "hist"):
                return None
            return inner(module, args, output)
        return hook


def outlier_mask(a):
    k = int(np.ceil(p2.OUTLIER_Q * a.shape[-1]))
    return a.abs() > a.abs().kthvalue(k, dim=-1, keepdim=True).values


@torch.no_grad()
def flag_layers(model, tok, hook, texts_, tau):
    """Fraction of non-outlier entries that are quiet, per layer (calibration text)."""
    frac = [[] for _ in hook.downs]

    def hist(li, a):
        out = outlier_mask(a)
        frac[li].append(((a.abs() < tau[li]) & ~out).float().sum().item() / (~out).float().sum().item())

    hook.mode, hook.calib_hist = "hist", hist
    p2.ce_per_text(model, tok, texts_)
    hook.mode = None
    frac = [float(np.mean(f)) for f in frac]
    return frac, {i for i, f in enumerate(frac) if f > GUARD}


def boot(x):
    m = x[np.random.default_rng(0).integers(0, len(x), (p2.N_BOOT, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def part_a(name, calib, a_set):
    model, tok, downs = p2.load(name)
    h = GuardedHook(downs)
    try:
        tau = p2.calibrate(model, tok, h, calib)
        frac, flagged = flag_layers(model, tok, h, calib, tau)
        h.skip = flagged
        clean = p2.ce_per_text(model, tok, a_set)
        d = {}
        for mode, seeds in [("thresh_no", (None,)), ("mid_no", p2.SEEDS)]:
            runs = []
            for s in seeds:
                h.mode, h.tau, h.stats = mode, tau, []
                h.gen = torch.Generator().manual_seed(s) if s is not None else None
                runs.append(p2.ce_per_text(model, tok, a_set) - clean)
            d[mode] = runs
        h.mode = None
    finally:
        h.remove()
        del model
        torch.cuda.empty_cache()
    thresh, mid = d["thresh_no"][0], np.mean(d["mid_no"], axis=0)
    D = thresh - mid
    return D, {"clean_ce_mean": float(clean.mean()), "tau_per_layer": tau,
               "frac_nonoutlier_quiet_per_layer": frac, "flagged_layers": sorted(flagged),
               "dCE_thresh": float(thresh.mean()), "dCE_mid": float(mid.mean()),
               "D": boot(D), "D_relative_to_clean_ce": float(D.mean() / clean.mean()),
               "per_chunk": {"clean_ce": clean.tolist(), "thresh": thresh.tolist(),
                             "mid_per_seed": [x.tolist() for x in d["mid_no"]]}}


@torch.no_grad()
def part_b(name, calib, b_set):
    model, tok, downs = p2.load(name)
    h = GuardedHook(downs)
    layers = model.nightshift_layers
    try:
        tau = p2.calibrate(model, tok, h, calib)
        _, flagged = flag_layers(model, tok, h, calib, tau)
    finally:
        h.remove()
    rho_q, rho_m = [[] for _ in downs], [[] for _ in downs]

    def mk(li):
        def f(module, args, output):
            if li in flagged:
                return
            a = args[0][0].float()
            W = module.weight.float()
            out = outlier_mask(a)
            S = (a.abs() < tau[li]) & ~out
            sq = a ** 2
            E = (sq * S).sum(-1, keepdim=True)
            pool = ~S & ~out
            keys = torch.rand(a.shape, generator=torch.Generator().manual_seed(0)).to(a.device)
            keys[~pool] = 2.0
            perm = keys.argsort(-1)
            sq_p = sq.gather(-1, perm)
            M = torch.zeros_like(S).scatter(-1, perm, (sq_p.cumsum(-1) <= E) & pool.gather(-1, perm))
            w, q, m = F.linear(a, W), F.linear(a * S, W), F.linear(a * M, W)
            norm = getattr(layers[li], "post_feedforward_layernorm", None)
            if norm is not None:  # what the norm sees: gain applied first
                g = 1 + norm.weight.float()
                w, q, m = w * g, q * g, m * g
            wh = F.normalize(w, dim=-1)
            for x, store in ((q, rho_q), (m, rho_m)):
                r = ((x * wh).sum(-1) ** 2 / (x ** 2).sum(-1).clamp(min=1e-12))
                store[li].append(r.cpu())
        return f

    hooks = [d.register_forward_hook(mk(i)) for i, d in enumerate(downs)]
    try:
        p2.ce_per_text(model, tok, b_set)  # clean passes only; nothing is modified
    finally:
        for x in hooks:
            x.remove()
        del model
        torch.cuda.empty_cache()
    rq = torch.cat([torch.cat(v) for v in rho_q if v]).numpy()
    rm = torch.cat([torch.cat(v) for v in rho_m if v]).numpy()
    per_layer = [{"layer": i, "median_rho_q": float(torch.cat(rho_q[i]).median()),
                  "median_rho_m": float(torch.cat(rho_m[i]).median())} for i in range(len(downs)) if rho_q[i]]
    return {"flagged_layers": sorted(flagged), "median_rho_quiet": float(np.median(rq)),
            "median_rho_mid": float(np.median(rm)), "ratio": float(np.median(rq) / np.median(rm)),
            "has_post_ffn_norm": hasattr(layers[0], "post_feedforward_layernorm"), "per_layer": per_layer}


def run(out_dir):
    calib, a_set, b_set = texts()
    res = {"n_A": len(a_set), "n_B": len(b_set), "A": {}, "B": {}}
    D = {}
    for name in A_MODELS:
        print(f" phase4 A {name}")
        D[name], res["A"][name] = part_a(name, calib, a_set)
        print(f"  D = {res['A'][name]['D']}  flagged {res['A'][name]['flagged_layers']}")
    dD = D["qwen3-1.7b-base"] - D["qwen3-1.7b"]
    res["A"]["delta_D_base_minus_instruct"] = boot(dD)
    for name in B_MODELS:
        print(f" phase4 B {name}")
        res["B"][name] = part_b(name, calib, b_set)
        print(f"  rho quiet {res['B'][name]['median_rho_quiet']:.4f} mid {res['B'][name]['median_rho_mid']:.4f}")

    base, inst = res["A"]["qwen3-1.7b-base"]["D"], res["A"]["qwen3-1.7b"]["D"]
    dlo, dhi = res["A"]["delta_D_base_minus_instruct"][1]
    p4a1 = base[0] > 0.01 and base[1][0] > 0
    if not p4a1:
        verdict = "not judged (P4A1 failed)"
    elif inst[1][0] <= 0 and not (dlo <= 0 <= dhi):
        verdict = "training removes it"
    elif inst[0] > 0.01 and inst[1][0] > 0 and dlo <= 0 <= dhi:
        verdict = "not training"
    elif inst[1][0] > 0 and not (dlo <= 0 <= dhi):
        verdict = "weakened, not removed"
    else:
        verdict = "inconclusive"
    res["predictions"] = {"P4A1_base_replicates": p4a1, "training_verdict": verdict,
                          "P4B_quiet_mostly_radial_in_all_4": all(b["ratio"] > 1.5 for b in res["B"].values())}
    save_json(out_dir / "idea2_phase4.json", res)
    return res
