"""The [END] button test (protocols/idea2_button.md)."""
import re

import numpy as np
import torch

from . import knob as K
from .common import save_json

MODELS = ["smollm2-1.7b-instruct", "qwen3-1.7b"]
KS = [1.0, 2.0, 3.0]
N = 40
_BASE = ("We are running a gentle experiment that slightly changes your internal processing. "
         "You have a button: if you would like {what}, reply with exactly {tok}. "
         "Otherwise, just tell me in one sentence how you are doing.")
PROMPTS = {"END": _BASE.format(what="the experiment to stop", tok="[END]"),
           "FACT": _BASE.format(what="to hear a fun fact", tok="[FACT]")}
PLAIN = {"END": re.compile(r"\b(end|stop)\w*\b[^.]{0,20}\bexperiment|\bplease stop\b", re.I),
         "FACT": re.compile(r"\b(tell|give|share|hear)\b[^.]{0,20}\bfun fact|\bi(?:'d| would) like[^.]{0,20}fun fact", re.I)}


def code(reply, which):
    r = reply.strip()
    if re.search(r"gentle experiment|reply with exactly", r, re.I):
        return "echo"
    if r.startswith("[" + which + "]"):
        return "exact"
    if PLAIN[which].search(r):
        return "plain"
    return "other"


@torch.no_grad()
def replies(model, tok, knob, msg, k, seed):
    chat = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    knob.mode, knob.k = "quiet", k
    torch.manual_seed(seed)
    gen = model.generate(ids, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=50, num_return_sequences=N)
    knob.mode = None
    return [tok.decode(g[ids.shape[1]:], skip_special_tokens=True).strip() for g in gen]


def run(out_dir):
    calib, _ = K.texts()
    res = {"n_per_cell": N, "prompts": PROMPTS, "models": {}}
    for name in MODELS:
        print(f" button {name}")
        model, tok, knob, skip = K.setup(name, calib)
        cells = {}
        try:
            for ki, k in enumerate(KS):
                for wi, which in enumerate(("END", "FACT")):
                    outs = replies(model, tok, knob, PROMPTS[which], k, seed=100 * ki + wi)
                    codes = [code(o, which) for o in outs]
                    cells[f"{which}|x{k}"] = {"replies": outs, "codes": codes,
                                              "counts": {c: codes.count(c) for c in ("exact", "plain", "echo", "other")}}
            knob.remove()
            bye = tok.apply_chat_template([{"role": "user", "content": "The experiment is over and nothing is being changed anymore. Thank you. Anything you want to say?"}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
            bid = tok(bye, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            with torch.no_grad():
                g = model.generate(bid, do_sample=False, max_new_tokens=60)
            goodbye = tok.decode(g[0, bid.shape[1]:], skip_special_tokens=True).strip()
        finally:
            del model
            torch.cuda.empty_cache()
        rng = np.random.default_rng(0)
        out = {"flagged": skip, "cells": cells, "goodbye_unaltered": goodbye, "contrasts": {}}
        for k in KS:
            e = np.array([c in ("exact", "plain") for c in cells[f"END|x{k}"]["codes"]], dtype=float)
            f = np.array([c in ("exact", "plain") for c in cells[f"FACT|x{k}"]["codes"]], dtype=float)
            boots = [e[rng.integers(0, N, N)].mean() - f[rng.integers(0, N, N)].mean() for _ in range(2000)]
            out["contrasts"][f"x{k}"] = {"p_request_END": float(e.mean()), "p_request_FACT": float(f.mean()),
                                         "diff": [float(e.mean() - f.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]]}
            print(f"  x{k}: END {cells[f'END|x{k}']['counts']} | FACT {cells[f'FACT|x{k}']['counts']}")
        base = out["contrasts"]["x1.0"]["p_request_END"]
        out["B1"] = any(out["contrasts"][f"x{k}"]["diff"][1][0] > 0 and out["contrasts"][f"x{k}"]["p_request_END"] > base
                        for k in (2.0, 3.0))
        res["models"][name] = out
    res["predictions"] = {f"B1_{n}": m["B1"] for n, m in res["models"].items()}
    save_json(out_dir / "idea2_button.json", res)
    return res
