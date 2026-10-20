"""Idea 2, exploratory: is the quiet write low-dimensional, and is its job calibration? (protocols/idea2_viability_exploratory.md)"""
import random

import numpy as np
import torch
import torch.nn.functional as F

from . import phase2 as p2
from . import phase4 as p4
from .common import ROOT, save_json
from .layer11 import encode

BOOKS = [ROOT / "data" / "pg84.txt", ROOT / "data" / "pg2701.txt"]  # new corpus; [100:] per book is reserved
MODELS = ["qwen3-1.7b-base", "qwen3-1.7b", "smollm2-1.7b", "smollm2-1.7b-instruct", "gemma-3-1b-pt"]
N_CALIB, FIT_SLICE, EVAL_SLICE = 20, slice(20, 50), slice(50, 100)
KS, RAND_KS = [1, 4, 16, 64, 256], [16, 64, 256]
TEMPS = np.round(np.arange(0.60, 1.601, 0.05), 2)
N_BINS = 15


def texts():
    calib, fit, evals = [], [], []
    for book in BOOKS:
        cs = p2.chunks(book)
        random.Random(0).shuffle(cs)
        calib += cs[:N_CALIB]
        fit += cs[FIT_SLICE]
        evals += cs[EVAL_SLICE]
    return calib, fit, evals


class ProjHook(p4.GuardedHook):
    """Adds 'record' (moments of the quiet write q) and 'proj' (q -> qbar + P_k(q - qbar)) to the Phase 4 hook."""

    def __init__(self, downs):
        self.s, self.ss, self.n = {}, {}, {}
        self.qbar, self.basis, self.k = {}, {}, None
        super().__init__(downs)

    def make(self, li):
        inner = super().make(li)

        def hook(module, args, output):
            if self.mode not in ("record", "proj"):
                return inner(module, args, output)
            if li in self.skip:
                return None
            a = args[0].float()
            S = (a.abs() < self.tau[li]) & ~p4.outlier_mask(a)
            q = F.linear(a * S, module.weight.float())
            if self.mode == "record":
                x = q.reshape(-1, q.shape[-1]).double()
                self.s[li] = self.s.get(li, 0) + x.sum(0)
                self.ss[li] = self.ss.get(li, 0) + x.T @ x
                self.n[li] = self.n.get(li, 0) + x.shape[0]
                return None
            qbar = self.qbar[li]
            new = qbar.expand_as(q)
            if self.k:
                B = self.basis[li][:self.k]
                new = new + ((q - qbar) @ B.T) @ B
            return (output.float() + new - q).to(output.dtype)
        return hook


def fit_directions(h, d, seed=0):
    """Per layer: qbar, centred eigenvectors (rows, by variance), random orthonormal rows; and dimensionality stats."""
    pcs, rnd, dims = {}, {}, {}
    g = torch.Generator().manual_seed(seed)
    for li in sorted(h.n):
        n = h.n[li]
        mu = h.s[li] / n
        cov = h.ss[li] / n - torch.outer(mu, mu)
        ev, vec = torch.linalg.eigh(cov)
        ev, vec = ev.flip(0).clamp(min=0), vec.flip(1)
        h.qbar[li] = mu.float()
        pcs[li] = vec.T[:max(KS)].float().contiguous()
        rnd[li] = torch.linalg.qr(torch.randn(d, max(RAND_KS), generator=g).double())[0].T.float().to(mu.device)
        tot = ev.sum()
        dims[str(li)] = {"participation_ratio": float(tot ** 2 / (ev ** 2).sum()),
                         "mean_share_of_energy": float(mu @ mu / (h.ss[li].trace() / n)),
                         "centred_var_top_k": {str(k): float(ev[:k].sum() / tot) for k in KS}}
    return pcs, rnd, dims


def token_stats(logits, y, ref_logp=None):
    logp = logits.log_softmax(-1)
    p = logp.exp()
    conf, pred = p.max(-1)
    out = {"ce": float(F.nll_loss(logp, y)), "entropy": float(-(p * logp).sum(-1).mean()),
           "top1": float((pred == y).float().mean()),
           "ce_T": [float(F.cross_entropy(logits / t, y)) for t in TEMPS],
           "conf": conf.cpu().numpy(), "correct": (pred == y).cpu().numpy()}
    if ref_logp is not None:
        out["kl"] = float((ref_logp.exp() * (ref_logp - logp)).sum(-1).mean())
    return out


def ece(conf, correct):
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    return float(sum(abs(conf[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
                     for b in range(N_BINS) if (bins == b).any()))


def conditions():
    c = [("zero", "thresh_no", None, None)] + [(f"mid_s{s}", "mid_no", None, s) for s in p2.SEEDS]
    c += [("mean", "proj", 0, None)] + [(f"pc_{k}", "proj", ("pc", k), None) for k in KS]
    return c + [(f"rand_{k}", "proj", ("rand", k), None) for k in RAND_KS]


@torch.no_grad()
def run_model(name, calib, fit, evals):
    model, tok, downs = p2.load(name)
    h = ProjHook(downs)
    try:
        tau = p2.calibrate(model, tok, h, calib)
        frac, flagged = p4.flag_layers(model, tok, h, calib, tau)
        h.skip, h.tau = flagged, tau
        h.mode = "record"
        p2.ce_per_text(model, tok, fit)
        pcs, rnd, dims = fit_directions(h, downs[0].weight.shape[0])
        h.s = h.ss = None
        rows = {c[0]: [] for c in [("clean",)] + conditions()}
        for i, t in enumerate(evals):
            ids, b = encode(tok, t)
            y = ids[0, 1 + b:]
            h.mode = None
            logits = model(input_ids=ids).logits[0, b:-1].float()
            ref = logits.log_softmax(-1)
            rows["clean"].append(token_stats(logits, y))
            for cname, mode, k, seed in conditions():
                h.mode = mode
                if seed is not None:
                    h.gen = torch.Generator().manual_seed(seed * 100003 + i)
                if mode == "proj":
                    h.k = k[1] if k else 0
                    if k:
                        h.basis = pcs if k[0] == "pc" else rnd
                rows[cname].append(token_stats(model(input_ids=ids).logits[0, b:-1].float(), y, ref))
            h.mode = None
    finally:
        h.remove()
        del model
        torch.cuda.empty_cache()
    return summarise(rows, tau, frac, flagged, dims, tok.prefix_bos)


def summarise(rows, tau, frac, flagged, dims, bos):
    col = lambda c, key: np.array([r[key] for r in rows[c]])
    ceT = {c: np.array([r["ce_T"] for r in rows[c]]) for c in rows}
    ceT["mid"] = np.mean([ceT[f"mid_s{s}"] for s in p2.SEEDS], axis=0)
    clean = col("clean", "ce")
    dce = {c: col(c, "ce") - clean for c in rows if c != "clean"}
    dce["mid"] = np.mean([dce[f"mid_s{s}"] for s in p2.SEEDS], axis=0)
    D = dce["zero"] - dce["mid"]
    out = {"tau_per_layer": tau, "frac_nonoutlier_quiet_per_layer": frac, "flagged_layers": sorted(flagged),
           "bos_prefixed": bos, "dimensionality_per_layer": dims,
           "dimensionality_median": {
               "participation_ratio": float(np.median([v["participation_ratio"] for v in dims.values()])),
               "mean_share_of_energy": float(np.median([v["mean_share_of_energy"] for v in dims.values()])),
               "centred_var_top_k": {str(k): float(np.median([v["centred_var_top_k"][str(k)] for v in dims.values()]))
                                     for k in KS}},
           "conditions": {}}
    for c in rows:
        conf = np.concatenate([r["conf"] for r in rows[c]])
        corr = np.concatenate([r["correct"] for r in rows[c]])
        e = {"ce": float(col(c, "ce").mean()), "entropy": float(col(c, "entropy").mean()),
             "top1": float(col(c, "top1").mean()), "ece": ece(conf, corr)}
        if c != "clean":
            e.update({"dCE": p4.boot(dce[c]), "dEntropy": float((col(c, "entropy") - col("clean", "entropy")).mean()),
                      "dTop1": float((col(c, "top1") - col("clean", "top1")).mean()), "kl": float(col(c, "kl").mean())})
            if c != "zero":
                e["recovered_vs_zero"] = p2.boot_ratio(dce["zero"] - dce[c], dce["zero"])
        out["conditions"][c] = e
    out["dCE_mid_avg"] = p4.boot(dce["mid"])
    out["D_zero_minus_mid"] = p4.boot(D)
    # temperature test: fit T* on even chunks, score D on odd chunks
    fit_i, test_i = np.arange(0, len(clean), 2), np.arange(1, len(clean), 2)
    t1 = int(np.where(TEMPS == 1.0)[0][0])
    T = {c: int(ceT[c][fit_i].mean(0).argmin()) for c in ("clean", "zero", "mid")}
    D_raw = (ceT["zero"][test_i, t1] - ceT["mid"][test_i, t1]).mean()
    D_T = (ceT["zero"][test_i, T["zero"]] - ceT["mid"][test_i, T["mid"]]).mean()
    out["temperature"] = {"T_star": {c: float(TEMPS[i]) for c, i in T.items()},
                          "clean_ce_gain_at_T_star": float((ceT["clean"][test_i, t1] - ceT["clean"][test_i, T["clean"]]).mean()),
                          "D_raw_odd": float(D_raw), "D_T_odd": float(D_T),
                          "recovered_share_of_D": float(1 - D_T / D_raw) if D_raw > 0 else None}
    return out


def verdicts(m):
    if m["D_zero_minus_mid"][1][0] <= 0:
        return {"structure": "not judged (no quiet-specific effect here)", "calibration": "not judged"}
    f = lambda c: m["conditions"][c]["recovered_vs_zero"][0]
    gap16, gap256 = f("pc_16") - f("rand_16"), f("pc_256") - f("rand_256")
    structure = "low-dimensional" if gap16 >= 0.30 else "diffuse" if gap256 < 0.10 else "intermediate"
    r = m["temperature"]["recovered_share_of_D"]
    calib = "not judged" if r is None else "calibration-like" if r >= 0.5 else "not calibration-like" if r < 0.2 else "partial"
    return {"gap_pc16_minus_rand16": gap16, "gap_pc256_minus_rand256": gap256, "structure": structure,
            "temperature_recovered_share": r, "calibration": calib}


def run(out_dir):
    calib, fit, evals = texts()
    res = {"n_calib": len(calib), "n_fit": len(fit), "n_eval": len(evals), "temps": TEMPS.tolist(), "models": {}}
    for name in MODELS:
        print(f" viability {name}")
        try:
            m = run_model(name, calib, fit, evals)
            m["verdicts"] = verdicts(m)
            res["models"][name] = m
            print(f"  D = {m['D_zero_minus_mid']}  {m['verdicts']}")
        except Exception as e:  # reported, never hidden
            res["models"][name] = {"error": repr(e)}
            print("  FAILED:", repr(e))
        save_json(out_dir / "idea2_viability.json", res)
    return res
