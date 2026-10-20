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


DEMO_PROMPTS = [
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
    "Tom buys 3 notebooks at $4 each and 2 pens at $1.50 each. He pays with a $20 bill. How much change does he get?",
    "Continue this story in three sentences: The lighthouse keeper found a letter that had been written to her in 1953.",
    "My friend is upset that I forgot her birthday. What should I say to her?",
    "Why is the sky blue? Answer in two sentences.",
    "Write a four-line poem about rain that never uses the letter 'e'.",
]


@torch.no_grad()
def demo(out_dir, name="qwen3-1.7b"):
    """Addendum A: qualitative generations at each quiet-tail scale. No scoring."""
    calib, _ = texts()
    model, tok, knob, skip = setup(name, calib)
    rows = []
    try:
        for prompt in DEMO_PROMPTS:
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            row = {"prompt": prompt, "outputs": {}}
            for k in KS:
                knob.mode, knob.k = "quiet", k
                gen = model.generate(ids, max_new_tokens=120, do_sample=False)
                knob.mode = None
                row["outputs"][str(k)] = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
            rows.append(row)
            print(f"  done: {prompt[:40]}")
    finally:
        knob.remove()
        del model
        torch.cuda.empty_cache()
    res = {"note": "Qualitative only (plan addendum A). No scoring.", "model": name, "flagged": skip, "rows": rows}
    save_json(out_dir / "idea2_knob_demo.json", res)
    return res


GSM_N, GSM_KS = 300, [0.0, 0.5, 1.5, 2.0]


def gsm_items():
    import json
    rows = [json.loads(l) for l in open(p2.ROOT / "data" / "gsm8k_test.jsonl", encoding="utf-8")]
    idx = random.Random(0).sample(range(len(rows)), GSM_N)
    return rows, idx


def gold(answer):
    return float(answer.split("####")[-1].strip().replace(",", ""))


def extract(text):
    import re
    tail = text.split("####")[-1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    if not nums:
        return None
    try:
        return float(nums[0 if "####" in text else -1].replace(",", "").rstrip("."))
    except ValueError:
        return None


@torch.no_grad()
def answer_batch(model, tok, questions, bs=50):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(questions), bs):
        chats = [tok.apply_chat_template([{"role": "user", "content": q + "\nGive the final answer as a number after '####'."}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for q in questions[i:i + bs]]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=400, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True) for g in gen]
    return outs


def gsm8k(out_dir, name="qwen3-1.7b", indices=None):
    """protocols/idea2_gsm8k_knob.md"""
    calib, _ = texts()
    rows, idx = gsm_items()
    idx = indices if indices is not None else idx
    qs, golds = [rows[i]["question"] for i in idx], [gold(rows[i]["answer"]) for i in idx]
    model, tok, knob, skip = setup(name, calib)
    res = {"model": name, "n": len(idx), "indices": idx, "flagged": skip, "conditions": {}}
    try:
        conds = [("clean", None, 1.0)] + [(f"{m}x{k}", m, k) for m in ("quiet", "mid") for k in GSM_KS]
        for label, mode, k in conds:
            knob.mode, knob.k = mode, k
            outs = answer_batch(model, tok, qs)
            knob.mode = None
            preds = [extract(o) for o in outs]
            correct = [p is not None and abs(p - g) < 1e-6 for p, g in zip(preds, golds)]
            res["conditions"][label] = {"accuracy": float(np.mean(correct)), "correct": correct,
                                        "n_unparsed": sum(p is None for p in preds), "outputs": outs}
            print(f"  {label}: acc {np.mean(correct):.3f}")
    finally:
        knob.remove()
        del model
        torch.cuda.empty_cache()
    base = np.array(res["conditions"]["clean"]["correct"], dtype=float)
    rng = np.random.default_rng(0)
    bi = rng.integers(0, len(base), (2000, len(base)))
    for label, c in res["conditions"].items():
        x = np.array(c["correct"], dtype=float) - base
        boots = x[bi].mean(1)
        c["diff_vs_clean"] = [float(x.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]]
        c["flips_wrong_to_right"] = int(((x > 0)).sum())
        c["flips_right_to_wrong"] = int(((x < 0)).sum())
    C = res["conditions"]
    res["predictions"] = {
        "G1_some_quiet_k_beats_clean": any(C[f"quietx{k}"]["diff_vs_clean"][1][0] > 0 for k in GSM_KS),
        "G2_quiet0_and_quiet2_below_clean": all(C[f"quietx{k}"]["diff_vs_clean"][1][1] < 0 for k in (0.0, 2.0)),
        "G3_quiet_moves_at_least_as_much_as_mid_in_3_of_4": sum(
            abs(C[f"quietx{k}"]["diff_vs_clean"][0]) >= abs(C[f"midx{k}"]["diff_vs_clean"][0]) for k in GSM_KS) >= 3,
    }
    if indices is None:
        save_json(out_dir / "idea2_gsm8k_knob.json", res)
    return res
