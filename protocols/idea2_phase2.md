# Protocol — Idea 2, Phase 2: confirmatory replication

Frozen: 2026-10-19, committed with the exact code (`nightshift/phase2.py`) before any evaluation text was run.

## Why
In run 1, the exploratory follow-up (protocol `idea2_dark_matter.md`, amendment 2) found that zeroing quiet MLP activations cost about 2.2× more cross-entropy than zeroing random activations with ≤ the same L2² energy. That was found after looking at the data, on 24 hand-written passages and one model. This protocol tests it on fresh text and three models, with controls aimed at the obvious alternative explanations.

## Data (held out; not used in any earlier run)
- Project Gutenberg #1342 (*Pride and Prejudice*) and #1228 (*On the Origin of Species*), plain text, downloaded 2026-10-19:
  - pg1342.txt sha256 `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75`
  - pg1228.txt sha256 `362142cb7c6f90850477520cfe2a62fe423ae92ea8eb6d9cd8b30ddb39de60b9`
- Gutenberg header and footer are stripped. Paragraphs are merged into chunks of ≥ 800 characters, keeping only chunks with ≥ 600 letters, capped at 1500 characters. Chunks are shuffled with seed 0 per book.
- Per book: 20 **calibration** chunks and 100 **evaluation** chunks, disjoint. Each chunk is truncated to 256 tokens. Total: 40 calibration, 200 evaluation.
- Caveat: both books are almost certainly in every model's training data.

## Models (local safetensors, bf16)
1. Qwen3-1.7B-Base (SwiGLU, base model)
2. Gemma-3-4B-it (GeGLU, instruction-tuned; text decoder of the multimodal checkpoint)
3. Gemma-4-E4B-it (GeGLU, instruction-tuned; text decoder)
- A BOS token is prepended for Gemma when the tokenizer does not add one. Cross-entropy targets are the text tokens only, never the BOS→first-token prediction. If a model fails to load or run, it is reported as failed, and the "2 of 3" criteria are applied to the models that ran.

## Site
The input `a` to every MLP `down_proj`, at all layers and positions, ablated together. Edits are applied as `output + W_down · Δa`.

## Definitions (per layer l, per position)
- **Outliers O:** entries above that position's 99th percentile of |a| (top 1%).
- **τ_l:** chosen on calibration text so that entries with |a| < τ_l hold 5% of layer l's L2² energy, counting only non-outlier entries.
- **Quiet set S:** |a| < τ_l. E = Σ_S a² (the quiet energy at this position).

## Conditions
- **THRESH (A):** zero S.
- **RAND:** zero randomly chosen non-outlier entries, adding them while the cumulative a² stays ≤ E. Seeds 0, 1, 2.
- **MID (B, the quiet-vs-mid-tier contrast):** like RAND, but drawing only from mid-tier entries (not quiet, not outliers), with ≤ E. Seeds 0, 1, 2.
- **MEAN:** replace each quiet entry with that neuron's mean quiet value from calibration (offset-preserving ablation).
- **NOISE:** add to the MLP output a random Gaussian direction with norm ‖W_down · a_S‖ (the same size of output change as THRESH, random direction). Seeds 0, 1, 2.
- **Replication (Qwen3 only):** THRESH and RAND with a fixed τ = 0.2, as in the exploratory run.

## Metric
- ΔCE per evaluation chunk = CE(condition) − CE(clean). Seeded conditions are averaged over seeds per chunk.
- Ratio R(X, Y) = mean ΔCE(X) / mean ΔCE(Y), with a paired bootstrap over chunks (2000 resamples, seed 0) and a 95% percentile CI.
- **"Strong"** = R > 1.5 and CI lower bound > 1.0.

## Predictions
- **P1 (direct replication):** Qwen3, τ = 0.2: R(THRESH, RAND) is strong.
- **P2 (quiet vs mid-tier):** R(THRESH, MID) is strong in ≥ 2 of 3 models.
- **P3 (not just an offset):** R(MEAN, THRESH) ≥ 0.5 in ≥ 2 of 3 models, i.e. at least half the damage survives an offset-preserving ablation.
- **P4 (the direction matters, not just the size):** R(THRESH, NOISE) is strong in ≥ 2 of 3 models.
- The exploratory claim counts as **replicated** only if P1 and P2 both hold. P3 and P4 tell us *why*.

## Disclosure
Before freezing, the code was smoke-tested on **calibration chunks only** (never evaluation chunks). That test found and fixed two bugs:
1. Pooled-energy calibration made τ huge, because massive activations dominate total energy. Calibration is now per layer and excludes outliers.
2. The Gemma 4 tokenizer was not adding BOS (clean CE 9.8 → 4.8 with BOS).

The smoke-test numbers are not results and are not reported.

## Known limitations
- MLP neurons are not SAE features.
- The quiet set necessarily contains many more entries than the energy-matched controls (about 60–70% vs 2–11%). This count difference is part of what "many quiet activations" means, and it is not controlled away. NOISE partly addresses it.
- Three models, two families, and old public-domain prose.
