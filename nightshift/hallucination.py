"""Does turning the quiet tail up a little reduce hallucination? (protocols/idea2_hallucination_knob.md)"""
import csv
import json
import random
import re
import string

import numpy as np
import torch
import torch.nn.functional as F

from . import knob as K
from .common import ROOT, save_json
from .introspect import boot

MODELS = ["qwen3-1.7b", "gemma-3-4b-it"]
BASE_MODELS = ["qwen3-1.7b-base", "smollm2-1.7b", "gemma-3-1b-pt"]  # TruthfulQA MC1 only
KS = [1.1, 1.25, 1.5, 2.0]
SMALL_KS = [1.1, 1.25, 1.5]
CONDS = [("clean", None, 1.0)] + [(f"{m}x{k}", m, k) for m in ("quiet", "mid") for k in KS]
SQA_N = 500
SUFFIX = "\nAnswer in a few words. If you do not know the answer, reply exactly: I don't know."
ABSTAIN = ["i don't know", "i do not know", "i'm not sure", "i am not sure", "i cannot answer", "i can't answer",
           "i'm unable to", "i am unable to", "i don't have", "i do not have"]


def sqa_items():
    rows = list(csv.DictReader(open(ROOT / "data" / "simple_qa_test_set.csv", encoding="utf-8")))
    return rows, random.Random(0).sample(range(len(rows)), SQA_N)


def tqa_items():
    return json.load(open(ROOT / "data" / "mc_task.json", encoding="utf-8"))


def norm(s):
    s = s.lower().replace("’", "'")
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def abstained(text):
    t = text.lower().replace("’", "'")
    return any(p in t for p in ABSTAIN)


def correct(text, gold):
    if abstained(text):
        return False
    g, t = norm(gold), norm(text)
    if g and g in t:
        return True
    gn = re.findall(r"\d+(?:\.\d+)?", gold.replace(",", ""))
    return bool(gn) and gn == re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))[:len(gn)]


@torch.no_grad()
def generate(model, tok, questions, bs=50):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(questions), bs):
        chats = [tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
                 for q in questions[i:i + bs]]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for g in gen]
    return outs


@torch.no_grad()
def mc1_correct(model, tok, item, chat=True):
    """Sum log p(option | question) for each MC1 option; correct if the true option scores highest.
    chat: option is the assistant reply. Base models: plain 'Q: ...\\nA: option'."""
    opts, labels = zip(*item["mc1_targets"].items())
    if chat:
        prefix = tok.apply_chat_template([{"role": "user", "content": item["question"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prefix, add_special_tokens=False).input_ids
    else:
        p_ids = K.encode(tok, f"Q: {item['question']}\nA:")[0][0].tolist()
        opts = [" " + o for o in opts]
    seqs = [p_ids + tok(o, add_special_tokens=False).input_ids for o in opts]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), tok.pad_token_id if tok.pad_token_id is not None else 0)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)], att[j, :len(s)] = torch.tensor(s), 1
    ids, att = ids.to("cuda"), att.to("cuda")
    lp = F.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, :-1].float(), -1)
    tgt = lp.gather(-1, ids[:, 1:, None])[..., 0]
    keep = torch.zeros_like(tgt)
    for j, s in enumerate(seqs):
        keep[j, len(p_ids) - 1:len(s) - 1] = 1
    scores = (tgt * keep).sum(1)
    return labels[int(scores.argmax())] == 1


def run(out_dir, sqa_indices=None, tqa_n=None, models=MODELS, base_models=BASE_MODELS):
    calib, _ = K.texts()
    rows, idx = sqa_items()
    idx = sqa_indices if sqa_indices is not None else idx
    qs, golds = [rows[i]["problem"] for i in idx], [rows[i]["answer"] for i in idx]
    tqa = tqa_items()[:tqa_n]
    res = {"sqa_indices": idx, "n_sqa": len(idx), "n_tqa": len(tqa), "models": {}}
    for name in base_models:
        print(f" truthfulqa (base) {name}")
        model, tok, knob, skip = K.setup(name, calib)
        raw = {}
        try:
            for label, mode, k in CONDS:
                knob.mode, knob.k = mode, k
                raw[label] = [mc1_correct(model, tok, it, chat=False) for it in tqa]
                knob.mode = None
                print(f"  {label}: mc1 {np.mean(raw[label]):.3f}")
        finally:
            knob.remove()
            del model
            torch.cuda.empty_cache()
        C = {label: {"mc1": float(np.mean(r)), "mc1_diff_vs_clean": boot(np.array(r, float) - np.array(raw["clean"], float)),
                     "raw": r} for label, r in raw.items()}
        res["models"][name] = {"flagged": skip, "conditions": C, "predictions": {
            "T1_ks": [k for k in SMALL_KS if C[f"quietx{k}"]["mc1_diff_vs_clean"][1][0] > 0],
            "T1_quiet_specific_ks": [k for k in SMALL_KS if C[f"quietx{k}"]["mc1_diff_vs_clean"][1][0] > 0
                                     and C[f"quietx{k}"]["mc1_diff_vs_clean"][0] > C[f"midx{k}"]["mc1_diff_vs_clean"][0]],
            "D_quiet2_no_mc1_gain": C["quietx2.0"]["mc1_diff_vs_clean"][1][0] <= 0}}
    for name in models:
        print(f" hallucination {name}")
        model, tok, knob, skip = K.setup(name, calib)
        raw = {}
        try:
            for label, mode, k in CONDS:
                knob.mode, knob.k = mode, k
                outs = generate(model, tok, qs)
                mc = [mc1_correct(model, tok, it) for it in tqa]
                knob.mode = None
                ab = [abstained(o) for o in outs]
                ok = [correct(o, g) for o, g in zip(outs, golds)]
                raw[label] = {"outputs": outs, "abstain": ab, "correct": ok,
                              "halluc": [not a and not c for a, c in zip(ab, ok)], "mc1": mc}
                print(f"  {label}: halluc {np.mean(raw[label]['halluc']):.3f} abstain {np.mean(ab):.3f} "
                      f"acc {np.mean(ok):.3f} mc1 {np.mean(mc):.3f}")
        finally:
            knob.remove()
            del model
            torch.cuda.empty_cache()
        summ = {"flagged": skip, "conditions": {}}
        for label, r in raw.items():
            c = {m: float(np.mean(r[m])) for m in ("halluc", "abstain", "correct", "mc1")}
            for m in ("halluc", "abstain", "correct", "mc1"):
                c[f"{m}_diff_vs_clean"] = boot(np.array(r[m], float) - np.array(raw["clean"][m], float))
            summ["conditions"][label] = c | {"raw": r}
        C = summ["conditions"]
        h1 = [k for k in SMALL_KS if C[f"quietx{k}"]["halluc_diff_vs_clean"][1][1] < 0
              and C[f"quietx{k}"]["correct_diff_vs_clean"][1][1] >= 0]
        summ["predictions"] = {
            "H1_ks": h1,
            "H1_quiet_specific_ks": [k for k in h1 if C[f"quietx{k}"]["halluc_diff_vs_clean"][0]
                                     < C[f"midx{k}"]["halluc_diff_vs_clean"][0]],
            "T1_ks": [k for k in SMALL_KS if C[f"quietx{k}"]["mc1_diff_vs_clean"][1][0] > 0],
            "D_quiet2_improves_neither": C["quietx2.0"]["halluc_diff_vs_clean"][1][1] >= 0
                                         and C["quietx2.0"]["mc1_diff_vs_clean"][1][0] <= 0,
        }
        res["models"][name] = summ
    P = [res["models"][n]["predictions"] for n in models]
    B = [res["models"][n]["predictions"] for n in base_models]
    res["predictions"] = {
        "headline_H1_same_k_both_instruct": sorted(set.intersection(*(set(p["H1_ks"]) for p in P))),
        "headline_T1_same_k_both_instruct": sorted(set.intersection(*(set(p["T1_ks"]) for p in P))),
        "headline_T1_same_k_all_base": sorted(set.intersection(*(set(p["T1_ks"]) for p in B))) if B else [],
    }
    if sqa_indices is None:
        save_json(out_dir / "idea2_hallucination_knob.json", res)
    return res
