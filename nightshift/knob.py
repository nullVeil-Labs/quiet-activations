"""Idea 2, exploratory: scale the quiet tail up and down (protocols/idea2_volume_knob_exploratory.md)."""
import random

import numpy as np
import torch
import torch.nn.functional as F

from . import phase2 as p2
from . import phase4 as p4
from .common import save_json

PAIRS = [("qwen3-1.7b-base", "qwen3-1.7b"), ("smollm2-1.7b", "smollm2-1.7b-instruct")]
KS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def texts():
    calib, evals = [], []
    for book in p2.BOOKS:
        cs = p2.chunks(book)
        random.Random(0).shuffle(cs)  # identical shuffle to phases 2-6
        calib += cs[:p2.N_CALIB]
        evals += cs[560:]
    return calib, evals


class Knob:
    """Forward hook on every down_proj: output += W @ ((k - 1) * a * mask), mask = quiet or energy-matched mid."""

    def __init__(self, downs, tau, skip):
        self.tau, self.skip, self.mode, self.k = tau, skip, None, 1.0
        self.hooks = [d.register_forward_hook(self.make(i)) for i, d in enumerate(downs)]

    def make(self, li):
        def hook(module, args, output):
            if self.mode is None or self.k == 1.0 or li in self.skip:
                return None
            a = args[0].float()
            out = p4.outlier_mask(a)
            S = (a.abs() < self.tau[li]) & ~out
            if self.mode == "quiet":
                mask = S
            else:
                sq = a ** 2
                E = (sq * S).sum(-1, keepdim=True)
                pool = ~S & ~out
                keys = torch.rand(a.shape, generator=torch.Generator().manual_seed(0)).to(a.device)
                keys[~pool] = 2.0
                perm = keys.argsort(-1)
                sq_p = sq.gather(-1, perm)
                mask = torch.zeros_like(S).scatter(-1, perm, (sq_p.cumsum(-1) <= E) & pool.gather(-1, perm))
            return (output.float() + F.linear((self.k - 1) * a * mask, module.weight.float())).to(output.dtype)
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()


def setup(name, calib):
    model, tok, downs = p2.load(name)
    h = p4.GuardedHook(downs)
    try:
        tau = p2.calibrate(model, tok, h, calib)
        _, skip = p4.flag_layers(model, tok, h, calib, tau)
    finally:
        h.remove()
    return model, tok, Knob(downs, tau, skip), sorted(skip)


@torch.no_grad()
def logprobs(model, tok, text):
    ids, b = encode(tok, text)
    return F.log_softmax(model(input_ids=ids).logits[0, b:-1].float(), -1), ids[0, 1 + b:]


def encode(tok, text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=p2.MAX_TOKENS).input_ids
    if tok.prefix_bos:
        ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], 1)
    b = int(tok.bos_token_id is not None and ids[0, 0].item() == tok.bos_token_id)
    return ids.to("cuda"), b


def run(out_dir):
    calib, evals = texts()
    res = {"n_eval": len(evals), "ks": KS, "pairs": {}}
    for base_name, inst_name in PAIRS:
        print(f" knob pair {base_name} / {inst_name}")
        mb, tb, kb, sb = setup(base_name, calib)
        mi, ti, ki, si = setup(inst_name, calib)
        out = {"flagged": {base_name: sb, inst_name: si}, base_name: {}, inst_name: {}, "kl_inst_to_base": {}}
        try:
            for mode in ("quiet", "mid"):
                for model_name, model, tok, knob in ((base_name, mb, tb, kb), (inst_name, mi, ti, ki)):
                    out[model_name][mode] = {}
                    for k in KS:
                        knob.mode, knob.k = mode, k
                        out[model_name][mode][str(k)] = float(np.mean(
                            [F.nll_loss(*logprobs(model, tok, t)).item() for t in evals]))
                        knob.mode = None
                # KL(p_base clean || p_instruct,k), same tokens (pairs share a tokenizer)
                out["kl_inst_to_base"][mode] = {}
                for k in KS:
                    kls = []
                    for t in evals:
                        kb.mode = None
                        lb, yb = logprobs(mb, tb, t)
                        ki.mode, ki.k = mode, k
                        li_, yi = logprobs(mi, ti, t)
                        ki.mode = None
                        assert torch.equal(yb, yi)
                        kls.append(F.kl_div(li_, lb, log_target=True, reduction="none").sum(-1).mean().item())
                    out["kl_inst_to_base"][mode][str(k)] = float(np.mean(kls))
                print(f"  {mode}: KL inst->base {out['kl_inst_to_base'][mode]}")
        finally:
            kb.remove()
            ki.remove()
            del mb, mi
            torch.cuda.empty_cache()
        for name in (base_name, inst_name):  # express CE as change vs clean
            clean = out[name]["quiet"]["1.0"]
            out[name] = {m: {k: v - clean for k, v in d.items()} for m, d in out[name].items()} | {"clean_ce": clean}
        res["pairs"][f"{base_name}|{inst_name}"] = out
    save_json(out_dir / "idea2_volume_knob.json", res)
    return res
