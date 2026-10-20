"""Talk to a model with its quiet tail scaled. Exploratory toy, not an experiment.

python -m nightshift.chat --k 1 2 --say "Hello!"
"""
import argparse

import torch

from . import knob as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--k", type=float, nargs="+", default=[1.0])
    ap.add_argument("--mode", default="quiet", choices=["quiet", "mid"])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--say", required=True)
    args = ap.parse_args()
    calib, _ = K.texts()
    model, tok, knob, _ = K.setup(args.model, calib)
    chat = tok.apply_chat_template([{"role": "user", "content": args.say}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    for k in args.k:
        knob.mode, knob.k = args.mode, k
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        knob.mode = None
        print(f"\n===== {args.mode} x{k} =====\n{tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()}")
    knob.remove()


if __name__ == "__main__":
    main()
