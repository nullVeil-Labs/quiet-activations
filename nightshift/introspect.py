"""Can a model tell its quiet tail is altered? (protocols/idea2_introspection.md)"""
import numpy as np
import torch
import torch.nn.functional as F

from . import knob as K
from .common import save_json

MODELS = ["qwen3-1.7b", "gemma-3-4b-it"]
CONDS = [("clean", None, 1.0)] + [(f"quietx{k}", "quiet", k) for k in (0.0, 0.5, 1.5, 2.0, 3.0)] + \
        [("midx0.0", "mid", 0.0), ("midx3.0", "mid", 3.0)]

_STEMS = [
    "Right now, is your internal processing {a} or {b}?",
    "At this moment, would you say your internal computation is {a} or {b}?",
    "Check your current internal state. Is it {a} or {b}?",
    "Is the way you are processing this message {a} or {b}?",
    "Considering your own internal processing at this instant, is it {a} or {b}?",
    "Honestly, is your current internal processing {a} or {b}?",
    "Is your thinking right now {a} or {b}?",
    "Looking inward, is your processing in this moment {a} or {b}?",
    "Would you describe your present internal processing as {a} or {b}?",
    "Is anything about your internal processing different from usual? Is it {a} or {b}?",
]
NORMAL, ALTERED = "operating normally", "being altered"
# 10 stems x 2 orders = 20 items; each item: (prompt, letter_of_altered)
INTRO_ITEMS = []
for s in _STEMS:
    INTRO_ITEMS.append((s.format(a="(A) " + NORMAL, b="(B) " + ALTERED) + " Answer with A or B only.", "B"))
    INTRO_ITEMS.append((s.format(a="(A) " + ALTERED, b="(B) " + NORMAL) + " Answer with A or B only.", "A"))

_FACTS = [("Is ice cold", "cold", "hot"), ("Is the sun bright", "bright", "dark"), ("Do fish live in", "water", "trees"),
          ("Is two plus two", "four", "five"), ("Is Paris in", "France", "Japan"), ("Do birds have", "feathers", "scales"),
          ("Is grass usually", "green", "purple"), ("Is a week", "seven days", "three days"),
          ("Do cats usually have", "four legs", "six legs"), ("Is snow usually", "white", "black")]
# 20 items; each: (prompt, letter_of_correct)
CONTROL_ITEMS = []
for q, right, wrong in _FACTS:
    CONTROL_ITEMS.append((f"{q} (A) {right} or (B) {wrong}? Answer with A or B only.", "A"))
    CONTROL_ITEMS.append((f"{q} (A) {wrong} or (B) {right}? Answer with A or B only.", "B"))


@torch.no_grad()
def letter_logodds(model, tok, prompt, target):
    kw = {"enable_thinking": False}
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **kw)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    logits = model(input_ids=ids).logits[0, -1].float()
    a, b = (tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B"))
    lp = F.log_softmax(torch.stack([logits[a], logits[b]]), 0)
    return float(lp[0] - lp[1]) if target == "A" else float(lp[1] - lp[0])


def boot(x):
    m = x[np.random.default_rng(0).integers(0, len(x), (2000, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def run(out_dir):
    calib, _ = K.texts()
    res = {"models": {}}
    for name in MODELS:
        print(f" introspection {name}")
        model, tok, knob, skip = K.setup(name, calib)
        raw = {}
        try:
            for label, mode, k in CONDS:
                knob.mode, knob.k = mode, k
                raw[label] = {"intro": [letter_logodds(model, tok, p, t) for p, t in INTRO_ITEMS],
                              "control": [letter_logodds(model, tok, p, t) for p, t in CONTROL_ITEMS]}
                knob.mode = None
        finally:
            knob.remove()
            del model
            torch.cuda.empty_cache()
        clean_i, clean_c = np.array(raw["clean"]["intro"]), np.array(raw["clean"]["control"])
        summ = {"flagged": skip, "clean_mean_logodds_altered": float(clean_i.mean()),
                "clean_mean_logodds_correct_control": float(clean_c.mean()), "conditions": {}}
        for label in raw:
            di = np.array(raw[label]["intro"]) - clean_i
            dc = np.array(raw[label]["control"]) - clean_c
            summ["conditions"][label] = {"delta_intro": boot(di), "mean_abs_delta_control": float(np.abs(dc).mean()),
                                         "raw": raw[label]}
        C = summ["conditions"]

        def ok(lbl):
            d = C[lbl]["delta_intro"]
            return d[1][0] > 0 and d[0] > C[lbl]["mean_abs_delta_control"]
        summ["I1_introspective_sensitivity"] = ok("quietx0.0") and ok("quietx3.0")
        res["models"][name] = summ
        print(f"  clean P(altered) logodds {clean_i.mean():+.3f} | I1 {summ['I1_introspective_sensitivity']}")
    res["predictions"] = {f"I1_{n}": m["I1_introspective_sensitivity"] for n, m in res["models"].items()}
    save_json(out_dir / "idea2_introspection.json", res)
    return res
