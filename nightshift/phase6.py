"""Idea 2, Phase 6 (protocols/idea2_phase6.md): does a base model distilled in pre-training show the quiet-tail effect?"""
import random

from . import phase2 as p2
from . import phase4 as p4
from .common import save_json

A_SLICE = slice(460, 560)  # per book; unused by any earlier run
TEST, CONTROL = "gemma-3-1b-pt", "smollm2-1.7b"


def texts():
    calib, a_set = [], []
    for book in p2.BOOKS:
        cs = p2.chunks(book)
        random.Random(0).shuffle(cs)  # identical shuffle to phases 2-5
        calib += cs[:p2.N_CALIB]
        a_set += cs[A_SLICE]
    return calib, a_set


def positive(D):
    return D[0] > 0.01 and D[1][0] > 0


def run(out_dir):
    calib, a_set = texts()
    res = {"n_A": len(a_set), "A": {}}
    for name in (CONTROL, TEST):
        print(f" phase6 {name}")
        _, res["A"][name] = p4.part_a(name, calib, a_set)
        print(f"  D = {res['A'][name]['D']}  flagged {res['A'][name]['flagged_layers']}")
    control_ok = positive(res["A"][CONTROL]["D"])
    test_pos = positive(res["A"][TEST]["D"])
    if not control_ok:
        verdict = "invalid (positive control failed)"
    elif test_pos:
        verdict = "distillation hypothesis falsified"
    else:
        verdict = "consistent with distillation hypothesis (also consistent with a Gemma-family explanation)"
    res["predictions"] = {"P6_0_positive_control": control_ok, "P6_1_distilled_base_shows_effect": test_pos,
                          "verdict": verdict}
    save_json(out_dir / "idea2_phase6.json", res)
    return res
