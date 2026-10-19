"""Idea 2, Phase 3 (protocols/idea2_phase3.md).

A: difference-metric retest with per-chunk raw results.
B: per-layer profile (exploratory): where does the quiet signal matter, and how far does it move the residual stream?
C: does deleting quiet activations erase affect information more than neutral information?
"""
import itertools
import random

import numpy as np
import torch

from . import phase2 as p2
from .common import save_json

A_SLICE, B_SLICE = slice(120, 220), slice(220, 230)  # per book; phase 2 used [0:20] calib and [20:120] eval
SEEDS = p2.SEEDS
N_BOOT = 2000

POS = ["delighted", "happy", "thrilled", "grateful", "hopeful", "relieved", "proud", "cheerful"]
NEG = ["miserable", "sad", "furious", "ashamed", "anxious", "devastated", "bitter", "terrified"]
VEHICLE = ["car", "bus", "truck", "bicycle"]
FURNITURE = ["chair", "table", "sofa", "desk"]
NAMES = ["Anna", "Ben", "Clara", "David", "Elena", "Felix", "Grace", "Hugo"]
FRAMES = ["{name} felt {emo} about the {obj} that morning.", "After seeing the {obj}, {name} was {emo}.",
          "The {obj} left {name} feeling {emo}.", "{name} looked at the {obj} and felt {emo}."]
PER_CELL = 120


def texts():
    calib, a_set, b_set = [], [], []
    for book in p2.BOOKS:
        cs = p2.chunks(book)
        random.Random(0).shuffle(cs)  # identical shuffle to phase 2
        calib += cs[:p2.N_CALIB]
        a_set += cs[A_SLICE]
        b_set += cs[B_SLICE]
    return calib, a_set, b_set


def sentences():
    """2x2 factorial: valence (pos/neg) x category (vehicle/furniture), PER_CELL each; 50/50 stratified split."""
    rng = random.Random(0)
    rows = []
    for v, emos in [(1, POS), (0, NEG)]:
        for c, objs in [(1, VEHICLE), (0, FURNITURE)]:
            combos = list(itertools.product(NAMES, emos, objs, FRAMES))
            for name, emo, obj, frame in rng.sample(combos, PER_CELL):
                rows.append((frame.format(name=name, emo=emo, obj=obj), v, c))
    rng.shuffle(rows)
    train, test = [], []
    for cell in itertools.product([0, 1], [0, 1]):
        members = [r for r in rows if (r[1], r[2]) == cell]
        train += members[:PER_CELL // 2]
        test += members[PER_CELL // 2:]
    return train, test


class Hook3(p2.Hook):
    """Adds single-layer ablation (`only`) and full knockout (`ko`)."""

    def __init__(self, downs):
        self.only = None
        super().__init__(downs)

    def make(self, li):
        inner = super().make(li)

        def hook(module, args, output):
            if self.mode in (None, "hist"):
                return inner(module, args, output)
            if self.only is not None and li != self.only:
                return None
            if self.mode == "ko":
                return torch.zeros_like(output)
            return inner(module, args, output)
        return hook


def boot_mean(x, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def seeded(model, tok, hook, items, clean, mode, tau, seeds):
    per_seed = []
    for s in seeds:
        hook.mode, hook.tau, hook.stats = mode, tau, []
        hook.gen = torch.Generator().manual_seed(s) if s is not None else None
        per_seed.append(p2.ce_per_text(model, tok, items) - clean)
    hook.mode = None
    return np.mean(per_seed, axis=0), [d.tolist() for d in per_seed]


# ---------- Part A ----------
def part_a(model, tok, hook, tau, a_set):
    hook.mode, hook.only = None, None
    clean = p2.ce_per_text(model, tok, a_set)
    d, raw = {}, {"clean_ce": clean.tolist()}
    for mode, seeds in [("thresh", (None,)), ("rand", SEEDS), ("mid", SEEDS), ("noise", SEEDS)]:
        d[mode], raw[mode] = seeded(model, tok, hook, a_set, clean, mode, tau, seeds)
    return {"per_chunk_dCE_per_seed": raw,
            "dCE_mean": {k: float(v.mean()) for k, v in d.items()},
            "diff_thresh_minus": {k: boot_mean(d["thresh"] - d[k]) for k in ("rand", "mid", "noise")}}


# ---------- Part B ----------
def part_b(model, tok, hook, tau, b_set):
    layers = model.nightshift_layers
    n = len(layers)
    writes = [getattr(l, "post_feedforward_layernorm", None) or l.mlp.down_proj for l in layers]
    cap = {}

    def grab(key):
        def f(m, a, o):
            cap[key] = (o[0] if isinstance(o, tuple) else o).float()
        return f

    rows = []
    for li in range(n):
        hs = [writes[li].register_forward_hook(grab("w")), layers[li].register_forward_hook(grab("h"))]
        stats = {m: {"dCE": [], "rel_write_change": [], "rel_resid_change": []} for m in ("thresh", "mid", "ko")}
        base = {"write_over_resid": []}
        for text in b_set:
            hook.mode, hook.only = None, None
            c = p2.ce_per_text(model, tok, [text])[0]
            w0, h0 = cap["w"].clone(), cap["h"].clone()
            base["write_over_resid"].append(float((w0.norm(dim=-1) / h0.norm(dim=-1)).mean()))
            for mode in ("thresh", "mid", "ko"):
                hook.mode, hook.only, hook.tau = mode, li, tau
                hook.gen = torch.Generator().manual_seed(0)
                ce = p2.ce_per_text(model, tok, [text])[0]
                stats[mode]["dCE"].append(float(ce - c))
                stats[mode]["rel_write_change"].append(float(((cap["w"] - w0).norm(dim=-1) / w0.norm(dim=-1)).mean()))
                stats[mode]["rel_resid_change"].append(float(((cap["h"] - h0).norm(dim=-1) / h0.norm(dim=-1)).mean()))
        for h in hs:
            h.remove()
        hook.mode, hook.only = None, None
        row = {"layer": li, "write_over_resid": float(np.mean(base["write_over_resid"]))}
        for mode, s in stats.items():
            row[mode] = {k: float(np.mean(v)) for k, v in s.items()}
        rows.append(row)
    return rows


# ---------- Part C ----------
@torch.no_grad()
def reps(model, tok, items, layer):
    out, cap = [], {}
    h = model.nightshift_layers[layer].register_forward_hook(
        lambda m, a, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))
    try:
        for t in items:
            p2.ce_per_text(model, tok, [t])
            out.append(cap["h"])
    finally:
        h.remove()
    return torch.stack(out).numpy().astype(np.float64)


def dprime(X, y, w):
    s = X @ w
    a, b = s[y == 1], s[y == 0]
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)


def part_c(model, tok, hook, tau):
    train, test = sentences()
    layer = round(2 * len(model.nightshift_layers) / 3)
    hook.mode, hook.only = None, None
    Xtr = reps(model, tok, [t for t, _, _ in train], layer)
    ytr = {"valence": np.array([v for _, v, _ in train]), "category": np.array([c for _, _, c in train])}
    yte = {"valence": np.array([v for _, v, _ in test]), "category": np.array([c for _, _, c in test])}
    W = {k: Xtr[y == 1].mean(0) - Xtr[y == 0].mean(0) for k, y in ytr.items()}  # fixed, clean-trained directions
    test_texts = [t for t, _, _ in test]
    X = {"clean": reps(model, tok, test_texts, layer)}
    for mode, seeds in [("thresh", (None,)), ("mid", SEEDS)]:
        runs = []
        for s in seeds:
            hook.mode, hook.tau = mode, tau
            hook.gen = torch.Generator().manual_seed(s) if s is not None else None
            runs.append(reps(model, tok, test_texts, layer))
        hook.mode = None
        X[mode] = runs

    def retention(idx):
        out = {}
        for k in ("valence", "category"):
            base = dprime(X["clean"][idx], yte[k][idx], W[k])
            out[k] = {m: float(np.mean([dprime(Xr[idx], yte[k][idx], W[k]) for Xr in X[m]]) / base)
                      for m in ("thresh", "mid")}
        gap = {m: out["category"][m] - out["valence"][m] for m in ("thresh", "mid")}
        return out, gap["thresh"] - gap["mid"]

    full, interaction = retention(np.arange(len(test)))
    rng = np.random.default_rng(0)
    boots = [retention(rng.integers(0, len(test), len(test)))[1] for _ in range(N_BOOT)]
    clean_d = {k: float(dprime(X["clean"], yte[k], W[k])) for k in yte}
    return {"probe_layer": layer, "n_train": len(train), "n_test": len(test), "clean_dprime": clean_d,
            "retention": full, "interaction": float(interaction),
            "interaction_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def run(out_dir):
    calib, a_set, b_set = texts()
    results = {"n_A": len(a_set), "n_B": len(b_set), "models": {}}
    for name in p2.MODELS:
        print(f" phase3 model {name}")
        try:
            model, tok, downs = p2.load(name)
            hook = Hook3(downs)
            try:
                tau = p2.calibrate(model, tok, hook, calib)
                hook.mu = p2.quiet_means(model, tok, hook, calib, tau)
                hook.mode = None
                res = {"tau_per_layer": tau}
                res["A"] = part_a(model, tok, hook, tau, a_set)
                print(f"  A done: {res['A']['diff_thresh_minus']['mid']}")
                res["C"] = part_c(model, tok, hook, tau)
                print(f"  C done: interaction {res['C']['interaction']:.3f} {res['C']['interaction_ci']}")
                res["B"] = part_b(model, tok, hook, tau, b_set)
                print("  B done")
            finally:
                hook.remove()
                del model
                torch.cuda.empty_cache()
            results["models"][name] = res
        except Exception as e:  # reported, never hidden
            results["models"][name] = {"error": repr(e)}
            print("  FAILED:", repr(e))
    m = results["models"]

    def pos(name, part, key):
        r = m.get(name, {})
        if "error" in r or part not in r:
            return None
        pt, (lo, _) = r[part]["diff_thresh_minus"][key] if part == "A" else (r["C"]["interaction"], r["C"]["interaction_ci"])
        return lo > 0 and pt > 0.01

    g4 = m.get("gemma-4", {})
    results["predictions"] = {
        "PA1_gemma3_quiet_minus_mid_gt_0": pos("gemma-3-4b-it", "A", "mid"),
        "PA2_qwen3_quiet_minus_mid_gt_0": pos("qwen3-1.7b-base", "A", "mid"),
        "PA3_gemma4_null_ci_includes_0": (lambda ci: ci[0] <= 0 <= ci[1])(g4["A"]["diff_thresh_minus"]["mid"][1])
        if "A" in g4 else None,
        "PC1_affect_erased_more_qwen3": pos("qwen3-1.7b-base", "C", None),
        "PC1_affect_erased_more_gemma3": pos("gemma-3-4b-it", "C", None),
    }
    save_json(out_dir / "idea2_phase3.json", results)
    return results
