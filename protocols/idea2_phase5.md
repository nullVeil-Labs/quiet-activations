# Protocol — Idea 2, Phase 5: does the base-vs-instruct result hold in a second family?

Frozen: 2026-10-20, committed with the exact code (`nightshift/phase5.py`, reusing `phase4.part_a` and `phase4.training_verdict`) before any Phase 5 evaluation data was run.

## Background
In Phase 4, Qwen3-1.7B-Base showed the quiet-activation effect (D = +0.040 [0.037, 0.044]). The same model after post-training did not (+0.008 [−0.003, 0.023]; paired drop +0.032 [0.019, 0.043]). Is this specific to Qwen, or a pattern?

## Models (chosen before any run)
- **HuggingFaceTB/SmolLM2-1.7B** (base), revision `effd688a12921b4cc83e3312b6feb579f70f9c71`. Llama architecture, 24 layers, no post-FFN norm, Apache-2.0. Local folder `models/SmolLM2-1b-base`.
- **HuggingFaceTB/SmolLM2-1.7B-Instruct**, revision `31b70e2e869a7173562077fd711b654946d38674`. Its model card lists the base above as `base_model`.
- Raw text, no chat template. The tokenizer does not add BOS, so the CE targets are the same as for Qwen.
- Before running, both models are checked to load with no missing or unexpected weights, and the safetensors file sizes are checked against Hugging Face.

## Method
**Identical to Phase 4 Part A:**
- outlier-excluded quiet set;
- per-layer τ_l (5% of non-outlier energy, the same 40 calibration chunks);
- the > 95% guard;
- MID seeds 0–2;
- per-chunk D = ΔCE(quiet) − ΔCE(MID);
- paired ΔD;
- 2000-resample bootstrap.

Data: Gutenberg chunks **[360:460] per book** (200 chunks), with the same seed-0 shuffle. No earlier run used them.

## Predictions
- **P5.1:** base D > 0.01 with CI lower bound > 0.
- **Training verdict:** Phase 4's pre-registered rule, applied unchanged ("training removes it" / "not training" / "weakened, not removed" / "inconclusive").
- **Interpretation, fixed in advance:**
  - If P5.1 holds and the verdict is "training removes it" or "weakened, not removed", the Qwen pattern replicates in a second family.
  - If P5.1 fails, the base-model effect is not general, and Phase 4's result may be Qwen-specific.
  - In every case, both models are reported.

## Limitations
- Two families, both around 1.7B.
- Raw-text CE is not the instruct model's training objective.
- Neurons, not SAE features.
