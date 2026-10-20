"""Idea 2, Phase 2: pre-registered replication with energy-matched controls (protocols/idea2_phase2.md)."""
import math
import random
import re

import numpy as np
import torch
import torch.nn.functional as F

from .common import ROOT, save_json

MODELS = {
    "qwen3-1.7b-base": ROOT / "models" / "qwen3-1.7b-base",
    "gemma-3-4b-it": ROOT / "models" / "gemma-3-4b-it",
    "gemma-4": ROOT / "models" / "gemma-4",  # google/gemma-4-E4B-it
}
EXTRA_MODELS = {  # used by later phases, not by phase 2's run()
    "qwen3-1.7b": ROOT / "models" / "qwen3-1.7b",
    "smollm2-1.7b": ROOT / "models" / "SmolLM2-1b-base",  # HuggingFaceTB/SmolLM2-1.7B (base), folder name as downloaded
    "smollm2-1.7b-instruct": ROOT / "models" / "smollm2-1.7b-instruct",
    "gemma-3-1b-pt": ROOT / "models" / "gemma-3-1b-pt",  # google/gemma-3-1b-pt (base, distilled in pre-training)
}
BOOKS = [ROOT / "data" / "pg1342.txt", ROOT / "data" / "pg1228.txt"]
N_CALIB, N_EVAL, MAX_TOKENS = 20, 100, 256  # per book
ENERGY_FRAC = 0.05
OUTLIER_Q = 0.99
SEEDS = [0, 1, 2]
N_BOOT = 2000


# ---------- corpus ----------
def chunks(path):
    text = path.read_text(encoding="utf-8")
    text = text.split("*** START OF", 1)[1].split("\n", 1)[1].split("*** END OF", 1)[0]
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    out, cur = [], ""
    for p in paras:
        cur = f"{cur} {p}".strip()
        if len(cur) >= 800:
            if sum(c.isalpha() for c in cur) >= 600:
                out.append(cur[:1500])
            cur = ""
    return out


def corpus():
    calib, evals = [], []
    for book in BOOKS:
        cs = chunks(book)
        random.Random(0).shuffle(cs)
        calib += cs[:N_CALIB]
        evals += cs[N_CALIB:N_CALIB + N_EVAL]
    return calib, evals


# ---------- models ----------
def load(name):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    path = MODELS.get(name) or EXTRA_MODELS[name]
    cfg = AutoConfig.from_pretrained(path)
    cls = AutoModelForImageTextToText if "ConditionalGeneration" in cfg.architectures[0] else AutoModelForCausalLM
    model = cls.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(path)
    n_layers = getattr(cfg, "text_config", cfg).num_hidden_layers
    found = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) == n_layers and hasattr(m[0], "mlp")
             and hasattr(m[0].mlp, "down_proj")]
    found.sort(key=lambda nm: "language" not in nm[0])  # prefer the text decoder
    layers = found[0][1]
    # Gemma needs BOS and its tokenizer here does not add it; Qwen base uses no BOS.
    bos = tok.bos_token_id
    tok.prefix_bos = "gemma" in name and bos is not None and tok("x").input_ids[0] != bos
    model.nightshift_layers = layers  # used by phase3; no effect on phase2
    return model, tok, [l.mlp.down_proj for l in layers]


# ---------- ablation hook ----------
class Hook:
    def __init__(self, downs):
        self.downs = downs
        self.mode, self.tau, self.mu, self.gen = None, None, None, None
        self.stats = []
        self.calib_hist = None
        self.hooks = [d.register_forward_hook(self.make(i)) for i, d in enumerate(downs)]

    def make(self, li):
        def hook(module, args, output):
            if self.mode is None:
                return None
            a = args[0].float()
            if self.mode == "hist":
                self.calib_hist(li, a)
                return None
            W = module.weight.float()
            sq = a ** 2
            quiet = a.abs() < (self.tau[li] if isinstance(self.tau, list) else self.tau)
            E = (sq * quiet).sum(-1, keepdim=True)
            k = math.ceil(OUTLIER_Q * a.shape[-1])
            cut = a.abs().kthvalue(k, dim=-1, keepdim=True).values
            outlier = a.abs() > cut
            if self.mode == "thresh":
                delta = -a * quiet
            elif self.mode == "mean":
                delta = (self.mu[li].to(a.device) - a) * quiet
            elif self.mode in ("rand", "mid"):
                pool = ~outlier if self.mode == "rand" else (~outlier & ~quiet)
                keys = torch.rand(a.shape, generator=self.gen).to(a.device)
                keys[~pool] = 2.0
                perm = keys.argsort(-1)
                sq_p = sq.gather(-1, perm)
                take_p = (sq_p.cumsum(-1) <= E) & pool.gather(-1, perm)
                take = torch.zeros_like(quiet).scatter(-1, perm, take_p)
                delta = -a * take
            elif self.mode == "noise":
                size = F.linear(a * quiet, W).norm(dim=-1, keepdim=True)
                n = torch.randn(output.shape, generator=self.gen).to(a.device).float()
                d_out = n / n.norm(dim=-1, keepdim=True) * size
                self.stats.append((float("nan"), float("nan")))
                return (output.float() + d_out).to(output.dtype)
            removed = (delta != 0)
            self.stats.append((removed.float().mean().item(), ((sq * removed).sum() / sq.sum()).item()))
            return (output.float() + F.linear(delta, W)).to(output.dtype)
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def ce_per_text(model, tok, texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).input_ids
        if tok.prefix_bos:
            ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], 1)
        b = int(tok.bos_token_id is not None and ids[0, 0].item() == tok.bos_token_id)
        ids = ids.to("cuda")
        logits = model(input_ids=ids).logits[0, b:-1].float()  # same targets with or without BOS
        out.append(F.cross_entropy(logits, ids[0, 1 + b:]).item())
    return np.array(out)


@torch.no_grad()
def calibrate(model, tok, hook, texts):
    """Per layer: tau_l so that |a| < tau_l holds ENERGY_FRAC of the layer's L2^2, outliers (top 1%/position) excluded."""
    edges = torch.logspace(-8, 4, 4001, device="cuda")
    energy = torch.zeros(len(hook.downs), 4000, device="cuda", dtype=torch.float64)

    def hist(li, a):
        k = math.ceil(OUTLIER_Q * a.shape[-1])
        cut = a.abs().kthvalue(k, dim=-1, keepdim=True).values
        x = a.abs()[a.abs() <= cut]
        idx = torch.bucketize(x, edges).clamp(1, 4000) - 1
        energy[li].index_add_(0, idx, x.double() ** 2)

    hook.mode, hook.calib_hist = "hist", hist
    ce_per_text(model, tok, texts)
    cum = energy.cumsum(1) / energy.sum(1, keepdim=True)
    return [float(edges[int((row >= ENERGY_FRAC).nonzero()[0]) + 1]) for row in cum]


@torch.no_grad()
def quiet_means(model, tok, hook, texts, tau):
    sums = [None] * len(hook.downs)
    counts = [None] * len(hook.downs)

    def hist(li, a):
        q = (a.abs() < tau[li]).float()
        s, c = (a * q).sum((0, 1)), q.sum((0, 1))
        sums[li] = s if sums[li] is None else sums[li] + s
        counts[li] = c if counts[li] is None else counts[li] + c

    hook.mode, hook.calib_hist = "hist", hist
    ce_per_text(model, tok, texts)
    return [s / c.clamp(min=1) for s, c in zip(sums, counts)]


def boot_ratio(num, den, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(N_BOOT, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def run_condition(model, tok, hook, texts, clean, mode, tau, seeds=(None,)):
    deltas, stats = [], []
    for s in seeds:
        hook.mode, hook.tau, hook.stats = mode, tau, []
        hook.gen = torch.Generator().manual_seed(s) if s is not None else None
        deltas.append(ce_per_text(model, tok, texts) - clean)
        stats.append(np.nanmean(np.array(hook.stats), axis=0).tolist() if hook.stats else None)
    d = np.mean(deltas, axis=0)
    return d, {"dCE_mean": float(d.mean()), "dCE_per_seed": [float(x.mean()) for x in deltas],
               "frac_entries_changed_and_L2sq_removed_per_seed": stats}


def run_model(name, calib, evals):
    model, tok, downs = load(name)
    hook = Hook(downs)
    try:
        tau = calibrate(model, tok, hook, calib)
        hook.mu = quiet_means(model, tok, hook, calib, tau)
        hook.mode = None
        clean = ce_per_text(model, tok, evals)
        res = {"model": name, "n_mlp_layers": len(downs), "tau_per_layer": tau, "bos_prefixed": tok.prefix_bos,
               "clean_ce_mean": float(clean.mean()), "conditions": {}}
        d = {}
        for mode, seeds in [("thresh", (None,)), ("rand", SEEDS), ("mid", SEEDS), ("mean", (None,)), ("noise", SEEDS)]:
            d[mode], res["conditions"][mode] = run_condition(model, tok, hook, evals, clean, mode, tau, seeds)
            print(f"  {name} {mode}: dCE {res['conditions'][mode]['dCE_mean']:.4f}")
        res["ratios"] = {
            "thresh_over_rand": boot_ratio(d["thresh"], d["rand"]),
            "thresh_over_mid": boot_ratio(d["thresh"], d["mid"]),
            "thresh_over_noise": boot_ratio(d["thresh"], d["noise"]),
            "mean_over_thresh": boot_ratio(d["mean"], d["thresh"]),
        }
        if name == "qwen3-1.7b-base":  # direct replication of the exploratory result at tau = 0.2
            t02, s02 = run_condition(model, tok, hook, evals, clean, "thresh", 0.2)
            r02, sr02 = run_condition(model, tok, hook, evals, clean, "rand", 0.2, SEEDS)
            res["replication_tau_0.2"] = {"thresh": s02, "rand": sr02, "thresh_over_rand": boot_ratio(t02, r02)}
    finally:
        hook.remove()
        del model
        torch.cuda.empty_cache()
    return res


def strong(ratio):
    point, (lo, _) = ratio
    return point > 1.5 and lo > 1.0


def run(out_dir):
    calib, evals = corpus()
    results = {"n_calib": len(calib), "n_eval": len(evals), "models": {}}
    for name in MODELS:
        print(f" phase2 model {name}")
        try:
            results["models"][name] = run_model(name, calib, evals)
        except Exception as e:  # reported, never hidden
            results["models"][name] = {"error": repr(e)}
            print("  FAILED:", repr(e))
    ok = {n: m for n, m in results["models"].items() if "error" not in m}
    q = ok.get("qwen3-1.7b-base")
    results["predictions"] = {
        "P1_replication_qwen_tau0.2": strong(q["replication_tau_0.2"]["thresh_over_rand"]) if q else None,
        "P2_quiet_vs_mid_in_at_least_2_of_3": sum(strong(m["ratios"]["thresh_over_mid"]) for m in ok.values()) >= 2,
        "P3_not_just_offset_in_at_least_2_of_3": sum(m["ratios"]["mean_over_thresh"][0] >= 0.5 for m in ok.values()) >= 2,
        "P4_direction_matters_in_at_least_2_of_3": sum(strong(m["ratios"]["thresh_over_noise"]) for m in ok.values()) >= 2,
    }
    save_json(out_dir / "idea2_phase2.json", results)
    return results
