"""Idea 2, Phase 5 (protocols/idea2_phase5.md): Phase 4's base-vs-instruct test on a second family (SmolLM2-1.7B)."""
import random

from . import phase2 as p2
from . import phase4 as p4
from .common import save_json

A_SLICE = slice(360, 460)  # per book; unused by any earlier run
PAIR = ("smollm2-1.7b", "smollm2-1.7b-instruct")


def texts():
    calib, a_set = [], []
    for book in p2.BOOKS:
        cs = p2.chunks(book)
        random.Random(0).shuffle(cs)  # identical shuffle to phases 2-4
        calib += cs[:p2.N_CALIB]
        a_set += cs[A_SLICE]
    return calib, a_set


def run(out_dir):
    calib, a_set = texts()
    res = {"n_A": len(a_set), "A": {}}
    D = {}
    for name in PAIR:
        print(f" phase5 {name}")
        D[name], res["A"][name] = p4.part_a(name, calib, a_set)
        print(f"  D = {res['A'][name]['D']}  flagged {res['A'][name]['flagged_layers']}")
    res["A"]["delta_D_base_minus_instruct"] = p4.boot(D[PAIR[0]] - D[PAIR[1]])
    ok, verdict = p4.training_verdict(res["A"][PAIR[0]]["D"], res["A"][PAIR[1]]["D"], res["A"]["delta_D_base_minus_instruct"])
    res["predictions"] = {"P5_1_base_shows_effect": ok, "training_verdict": verdict}
    save_json(out_dir / "idea2_phase5.json", res)
    return res
