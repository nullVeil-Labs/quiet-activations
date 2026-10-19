"""Shared helpers: run IDs, result saving, model loading."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "qwen3-1.7b-base"
RESULTS_DIR = ROOT / "results"


def new_run_dir(tag):
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{tag}"
    path = RESULTS_DIR / run_id
    path.mkdir(parents=True, exist_ok=False)  # never overwrite a run
    return path


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


_model_cache = {}


def load_model():
    """Load Qwen3-1.7B once per process. Returns (model, tokenizer)."""
    if "m" not in _model_cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not MODEL_DIR.exists():
            raise SystemExit(f"Model not found at {MODEL_DIR}. See README for download instructions.")
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _model_cache["m"] = (model, tok)
    return _model_cache["m"]
