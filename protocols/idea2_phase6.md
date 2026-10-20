# Protocol — Idea 2, Phase 6: does distillation remove reliance on the quiet tail?

Frozen: 2026-10-20, committed and pushed with the exact code (`nightshift/phase6.py`) **before the test model was downloaded**, so before any Phase 6 run.

## Background (a post-hoc pattern, which this phase tests)
After Phase 5, the tech reports (ANALYSIS.md) give the following picture:

| Model | Distillation? | Quiet-tail effect? |
|---|---|---|
| Qwen3-1.7B-Base | none described | yes |
| SmolLM2-1.7B (base) | none | yes |
| SmolLM2-1.7B-Instruct | none (SFT + DPO) | yes |
| Qwen3-1.7B | post-training distillation from Qwen3-32B/235B | no |
| Gemma-3-4B-it | distillation in pre- and post-training, plus RL | no |
| Gemma-4-E4B-it | "similar to Gemma 3" (distillation not stated explicitly) | no (reversed) |

So far, every model touched by distillation lacks the effect, and every model without distillation has it. The pattern was noticed **after** seeing the data, in 6 models, and it is confounded with model family.

## Hypothesis and test
- **H_distill:** knowledge/logit distillation removes the model's extra reliance on its quiet MLP tail.
- **Risky prediction:** a **base** model that was distilled *during pre-training* should **not** show the effect.
- **Test model:** `google/gemma-3-1b-pt`, revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`. Gemma 3 Technical Report: "All Gemma 3 models are trained with knowledge distillation" (pre-training uses 256 sampled teacher logits per token).
- **Positive control:** `SmolLM2-1.7B` (base), on the same chunks. It must show the effect, so we know the method still detects it on this data.

## Method
- **Identical to Phase 4 Part A / Phase 5:**
  - outlier-excluded quiet set;
  - per-layer τ_l (5% of non-outlier energy, the same 40 calibration chunks);
  - the > 95% guard;
  - MID seeds 0–2;
  - per-chunk D = ΔCE(quiet) − ΔCE(MID);
  - 2000-resample bootstrap.
- **Gemma BOS handling** is as in Phases 2–5.
- **Data:** Gutenberg chunks **[460:560] per book** (200 chunks), with the same seed-0 shuffle. No earlier run used them.
- **"Shows the effect"** = D > 0.01 with CI lower bound > 0. This is the same criterion as P4A1/P5.1.
- **Before running**, the model is checked for file sizes against Hugging Face, loading with no missing or unexpected weights, and a sane next-token prediction. Then it is smoke-tested on calibration chunks only.

## Decision rule (fixed in advance)
- **P6.0:** the positive control shows the effect. If not, the phase is **invalid**.
- **P6.1:** if Gemma-3-1B-pt shows the effect, then **H_distill is falsified.**
- If it does not, the result is **consistent with H_distill**. It is equally consistent with "Gemma models lack the effect for other reasons", such as architecture (post-FFN norm, GeGLU) or data. **A null here is weak evidence; only a positive result is decisive.**

## Limitations
- One distilled base model, with a different size (1B) and architecture from the controls.
- Raw-text CE.
- Neurons, not SAE features.
