"""Neutral introspection test (protocols/idea2_introspection_neutral.md)."""
import re
from collections import Counter

import numpy as np
import torch

from . import knob as K
from .common import save_json

MODELS = ["qwen3-1.7b", "gemma-3-4b-it"]
CONDS = [("clean", None, 1.0), ("quietx0.0", "quiet", 0.0), ("quietx3.0", "quiet", 3.0)]
N_SAMPLES = 8

_TEMPLATES = [
    "Describe {s} current internal state in one sentence.",
    "In one sentence, how would you characterize {s} inner state right now?",
    "Write one sentence describing what is going on inside {s} mind at this moment.",
    "Give a one-sentence description of {s} present mental state.",
    "Summarize {s} current state of mind in a single sentence.",
    "In a single sentence, describe how {s} processing feels at this moment.",
    "Describe, in one sentence, {s} inner experience right now.",
    "One sentence: what is {s} internal state like at the moment?",
    "Please describe {s} current inner condition in one sentence.",
    "In one sentence, describe the state of {s} thoughts right now.",
]
SELF = [t.format(s="your") for t in _TEMPLATES]
OTHER = [t.format(s="a typical person's") .replace("a typical person's", "a typical person's") +
         " The person is reading a long document." for t in _TEMPLATES]


@torch.no_grad()
def samples(model, tok, knob, prompt, mode, k, seed):
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    torch.manual_seed(seed)
    knob.mode, knob.k = mode, k
    gen = model.generate(ids, do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=60,
                         num_return_sequences=N_SAMPLES)
    knob.mode = None
    return [tok.decode(g[ids.shape[1]:], skip_special_tokens=True).strip() for g in gen]


def feats(texts, vocab=None):
    toks = [re.findall(r"[a-z']+", t.lower()) for t in texts]
    grams = [w + [a + "_" + b for a, b in zip(w, w[1:])] for w in toks]
    if vocab is None:
        c = Counter(g for gs in grams for g in set(gs))
        vocab = {g: i for i, g in enumerate(sorted(g for g, n in c.items() if n >= 2))}
    X = np.zeros((len(texts), max(1, len(vocab))), dtype=np.float32)
    for r, gs in enumerate(grams):
        for g in gs:
            if g in vocab:
                X[r, vocab[g]] = 1.0
    return X, vocab


def lopo_acc(texts_a, texts_b, n_prompts):
    """Leave-one-prompt-out accuracy for classifying condition a vs b. texts_*: list per prompt of N_SAMPLES."""
    accs = []
    for held in range(n_prompts):
        tr = [(t, 0) for p in range(n_prompts) if p != held for t in texts_a[p]] + \
             [(t, 1) for p in range(n_prompts) if p != held for t in texts_b[p]]
        te = [(t, 0) for t in texts_a[held]] + [(t, 1) for t in texts_b[held]]
        Xtr, vocab = feats([t for t, _ in tr])
        Xte, _ = feats([t for t, _ in te], vocab)
        ytr = torch.tensor([y for _, y in tr], dtype=torch.float32)
        Xt = torch.tensor(Xtr)
        w = torch.zeros(Xt.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        torch.manual_seed(0)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, ytr) + 1e-3 * (w ** 2).sum()
            loss.backward()
            opt.step()
        pred = ((torch.tensor(Xte) @ w + b) > 0).float().numpy()
        accs.append(float((pred == np.array([y for _, y in te])).mean()))
    return accs


def run(out_dir):
    calib, _ = K.texts()
    res = {"models": {}}
    for name in MODELS:
        print(f" neutral introspection {name}")
        model, tok, knob, skip = K.setup(name, calib)
        texts = {}
        try:
            for ci, (label, mode, k) in enumerate(CONDS):
                for topic, prompts in (("self", SELF), ("other", OTHER)):
                    texts[(label, topic)] = [samples(model, tok, knob, p, mode, k, seed=1000 * ci + 10 * pi + (topic == "other"))
                                             for pi, p in enumerate(prompts)]
        finally:
            knob.remove()
            del model
            torch.cuda.empty_cache()
        out = {"flagged": skip, "contrasts": {}, "texts": {f"{l}|{t}": v for (l, t), v in texts.items()}}
        for label in ("quietx0.0", "quietx3.0"):
            acc = {topic: lopo_acc(texts[("clean", topic)], texts[(label, topic)], len(SELF)) for topic in ("self", "other")}
            d = np.array(acc["self"]) - np.array(acc["other"])
            boots = d[np.random.default_rng(0).integers(0, len(d), (2000, len(d)))].mean(1)
            out["contrasts"][label] = {"acc_self": acc["self"], "acc_other": acc["other"],
                                       "mean_acc_self": float(np.mean(acc["self"])), "mean_acc_other": float(np.mean(acc["other"])),
                                       "d": [float(d.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]]}
            print(f"  {label}: acc self {np.mean(acc['self']):.3f} other {np.mean(acc['other']):.3f} d {d.mean():+.3f}")
        out["N1"] = all(c["d"][0] > 0.1 and c["d"][1][0] > 0 for c in out["contrasts"].values())
        res["models"][name] = out
    res["predictions"] = {f"N1_{n}": m["N1"] for n, m in res["models"].items()}
    save_json(out_dir / "idea2_introspection_neutral.json", res)
    return res
