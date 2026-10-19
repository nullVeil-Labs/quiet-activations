"""Idea 2, exploratory: what is in Gemma-3-4B-it layer 11? (protocols/idea2_layer11_exploratory.md)"""
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from . import phase2 as p2
from .common import save_json

MODEL, LAYER, SLICE = "gemma-3-4b-it", 11, slice(230, 250)


def texts():
    calib, evals = [], []
    for book in p2.BOOKS:
        cs = p2.chunks(book)
        random.Random(0).shuffle(cs)  # identical shuffle to phases 2-3
        calib += cs[:p2.N_CALIB]
        evals += cs[SLICE]
    return calib, evals


def encode(tok, text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=p2.MAX_TOKENS).input_ids
    if tok.prefix_bos:
        ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], 1)
    b = int(tok.bos_token_id is not None and ids[0, 0].item() == tok.bos_token_id)
    return ids.to("cuda"), b


@torch.no_grad()
def token_ce(model, tok, text):
    ids, b = encode(tok, text)
    logits = model(input_ids=ids).logits[0, b:-1].float()
    return F.cross_entropy(logits, ids[0, 1 + b:], reduction="none").cpu().numpy(), ids[0, 1 + b:].tolist()


class L11:
    """Forward hook on down_proj of every layer: records stats, and edits layer LAYER's quiet write q."""

    def __init__(self, downs, tau):
        self.downs, self.tau = downs, tau
        self.mode, self.qbar, self.V = None, None, None
        self.record = False
        self.split = defaultdict(list)    # per-layer cancellation stats (clean)
        self.q_rows, self.mid_rows, self.quiet_counts, self.quiet_energy, self.n_tok = [], [], None, None, 0
        self.hooks = [d.register_forward_hook(self.make(i)) for i, d in enumerate(downs)]

    def make(self, li):
        def hook(module, args, output):
            a = args[0][0].float()                       # (T, I)
            W = module.weight.float()
            S = a.abs() < self.tau[li]
            q = F.linear(a * S, W)                        # quiet write, (T, d)
            if self.record:
                w = output[0].float()
                self.split[li].append(torch.stack([
                    q.norm(dim=-1) / w.norm(dim=-1),
                    F.cosine_similarity(q, w, dim=-1),
                    w.norm(dim=-1) / (q.norm(dim=-1) + (w - q).norm(dim=-1)),
                ], -1).cpu())
                if li == LAYER:
                    self.q_rows.append(q.cpu())
                    self.mid_rows.append(self.mid_write(a, W, S).cpu())
                    c = S.sum(0).cpu()
                    e = (((a * S) ** 2).sum(0) * (W ** 2).sum(0)).cpu()  # per-neuron quiet write energy
                    self.quiet_counts = c if self.quiet_counts is None else self.quiet_counts + c
                    self.quiet_energy = e if self.quiet_energy is None else self.quiet_energy + e
                    self.n_tok += a.shape[0]
            if li != LAYER or self.mode is None:
                return None
            if self.mode == "zeroq":
                new = -q
            elif self.mode == "meanq":
                new = self.qbar.to(q.device) - q
            else:  # ("pcq", k): keep only q's projection onto the top-k calibration directions
                Vk = self.V[:self.mode[1]].to(q.device)
                new = (q @ Vk.T) @ Vk - q
            return (output.float() + new.unsqueeze(0)).to(output.dtype)
        return hook

    @staticmethod
    def mid_write(a, W, S, seed=0):
        """Mid-tier write of the same energy as the quiet set (Phase 2 MID rule, outliers excluded)."""
        sq = a ** 2
        E = (sq * S).sum(-1, keepdim=True)
        k = int(np.ceil(p2.OUTLIER_Q * a.shape[-1]))
        cut = a.abs().kthvalue(k, dim=-1, keepdim=True).values
        pool = ~S & (a.abs() <= cut)
        keys = torch.rand(a.shape, generator=torch.Generator().manual_seed(seed)).to(a.device)
        keys[~pool] = 2.0
        perm = keys.argsort(-1)
        sq_p = sq.gather(-1, perm)
        take = torch.zeros_like(S).scatter(-1, perm, (sq_p.cumsum(-1) <= E) & pool.gather(-1, perm))
        return F.linear(a * take, W)

    def reset_records(self):
        self.split.clear()
        self.q_rows, self.mid_rows, self.quiet_counts, self.quiet_energy, self.n_tok = [], [], None, None, 0

    def remove(self):
        for h in self.hooks:
            h.remove()


def direction_stats(rows):
    X = torch.cat(rows).double()
    s = torch.linalg.svdvals(X)
    frac = (s ** 2 / (s ** 2).sum()).tolist()
    mean_dir = X.mean(0) / X.mean(0).norm()
    return {"frac_energy_top1_3_10": [sum(frac[:k]) for k in (1, 3, 10)],
            "mean_cos_to_mean_direction": float((F.normalize(X, dim=-1) @ mean_dir).mean())}


def run(out_dir):
    calib, evals = texts()
    model, tok, downs = p2.load(MODEL)
    probe = p2.Hook(downs)
    try:
        tau = p2.calibrate(model, tok, probe, calib)
    finally:
        probe.remove()
    h = L11(downs, tau)
    res = {"model": MODEL, "layer": LAYER, "n_calib": len(calib), "n_eval": len(evals), "tau_layer": tau[LAYER]}
    try:
        # calibration: mean quiet write and top directions
        h.record = True
        for t in calib:
            token_ce(model, tok, t)
        Xc = torch.cat(h.q_rows).double()
        h.qbar = Xc.mean(0).float()
        h.V = torch.linalg.svd(Xc, full_matrices=False).Vh[:10].float()
        h.reset_records()

        # clean eval pass: records 1, 3, 5
        clean, ids_all = [], []
        for t in evals:
            ce, ids = token_ce(model, tok, t)
            clean.append(ce)
            ids_all.append(ids)
        h.record = False
        res["cancellation_per_layer"] = {
            str(li): dict(zip(["q_over_w", "cos_q_w", "w_over_sum_parts"],
                              torch.cat(v).median(0).values.tolist())) for li, v in sorted(h.split.items())}
        res["directions"] = {"quiet": direction_stats(h.q_rows), "mid": direction_stats(h.mid_rows)}
        frac_quiet = (h.quiet_counts.float() / h.n_tok)
        always = frac_quiet >= 0.99
        res["neurons"] = {"n_neurons": int(frac_quiet.numel()), "n_quiet_ge_99pct": int(always.sum()),
                          "n_quiet_ge_90pct": int((frac_quiet >= 0.9).sum()),
                          "median_frac_quiet": float(frac_quiet.median()),
                          "share_of_quiet_energy_from_ge_99pct_neurons": float(h.quiet_energy[always].sum() / h.quiet_energy.sum())}

        # ablations at layer 11 only
        conds = {"zeroq": "zeroq", "meanq": "meanq", "pcq1": ("pcq", 1), "pcq3": ("pcq", 3), "pcq10": ("pcq", 10)}
        per_tok = {}
        for name, mode in conds.items():
            h.mode = mode
            per_tok[name] = [token_ce(model, tok, t)[0] - c for t, c in zip(evals, clean)]
        h.mode = None
        res["ablation_dCE_mean"] = {k: float(np.mean([d.mean() for d in v])) for k, v in per_tok.items()}

        # 2. which tokens / positions
        d = per_tok["zeroq"]
        pos_first = float(np.mean(np.concatenate([x[:8] for x in d])))
        pos_rest = float(np.mean(np.concatenate([x[8:] for x in d])))
        by_tok = defaultdict(list)
        for x, ids in zip(d, ids_all):
            for v, i in zip(x, ids):
                by_tok[i].append(v)
        top = sorted(((np.mean(v), i, len(v)) for i, v in by_tok.items() if len(v) >= 5), reverse=True)[:20]
        res["tokens"] = {"dCE_first8_positions": pos_first, "dCE_rest": pos_rest,
                         "share_of_total_from_top1pct_tokens": float(
                             np.sort(np.concatenate(d))[::-1][:max(1, len(np.concatenate(d)) // 100)].sum()
                             / np.concatenate(d).sum()),
                         "top_target_tokens": [{"token": tok.decode([i]), "mean_dCE": float(m), "n": n} for m, i, n in top]}

        # 6. logit lens of the mean quiet write, through post-FFN norm gain and final norm gain
        layer = model.nightshift_layers[LAYER]
        g_post = 1 + layer.post_feedforward_layernorm.weight.float()
        lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
        g_fin = 1 + lm.norm.weight.float()
        direction = h.qbar.cuda() * g_post * g_fin
        logits = model.lm_head.weight.float() @ direction
        res["logit_lens_qbar"] = {"promoted": [tok.decode([i]) for i in logits.topk(15).indices.tolist()],
                                  "suppressed": [tok.decode([i]) for i in (-logits).topk(15).indices.tolist()]}
    finally:
        h.remove()
        del model
        torch.cuda.empty_cache()
    save_json(out_dir / "idea2_layer11.json", res)
    return res
