# Protocol — Idea 2, Phase 4: training or architecture?

Frozen: 2026-10-20, committed with the exact code (`nightshift/phase4.py`) before any Phase 4 evaluation data was run.

## Background
- After the outlier-artifact correction (ANALYSIS.md, 2026-10-20), the quiet-activation effect stands in Qwen3-1.7B-**Base** only. Gemma-3-4B-**it** shows no clear effect, and Gemma-4-E4B-**it** is reversed.
- That comparison confounds **training** (base vs instruction-tuned) with **architecture**. One notable architectural difference is that both Gemmas apply an RMSNorm to the MLP output (`post_feedforward_layernorm`) before adding it to the residual stream; Qwen3 does not.
- Part A tests training: Qwen3 base vs Qwen3 instruct, same architecture.
- Part B tests the norm idea **observationally**. By the author's choice, no Gemma forward pass is modified.

## Shared setup
- Site, outlier definition (top 1% |a| per position) and per-layer τ_l calibration (5% of non-outlier L2² energy, on the same 40 calibration chunks) are unchanged from Phases 2–3.
- **Quiet set (fixed definition):** S = {|a| < τ_l} **minus outliers** — definition (ii) from the 2026-10-20 artifact check.
- MID: random mid-tier entries (not quiet, not outliers) with cumulative a² ≤ E = Σ_S a². Seeds 0, 1, 2.
- **Guard:** a layer where more than 95% of non-outlier entries are quiet on calibration text is flagged. Flagged layers are not ablated in any condition, and they are reported.
- Data: Gutenberg chunks (same books, same seed-0 shuffle). **[250:350] per book for Part A** (200 chunks) and **[350:360] per book for Part B** (20 chunks). No earlier run used them.

## Part A — base vs instruct (confirmatory)
- Models: Qwen3-1.7B-Base, and Qwen3-1.7B (post-trained/instruct; HF `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`). Raw text, no chat template, no BOS (Qwen convention).
- Per chunk: D = ΔCE(THRESH) − ΔCE(MID), with MID averaged over seeds. Base and instruct use the same chunks, so ΔD = D_base − D_instruct is paired per chunk. 95% bootstrap CIs (2000 resamples, seed 0).
- Relative D (D / clean CE) is also reported, descriptively, because instruct models predict raw text less well.
- **P4A1 (replication):** base: D > 0.01 and CI lower bound > 0.
- **Verdict on training** (judged only if P4A1 holds):
  - **"Training removes it"** if the instruct D CI includes 0 or lies below 0, **and** the ΔD CI excludes 0.
  - **"Not training"** if instruct D > 0.01 with CI lower bound > 0, **and** the ΔD CI includes 0.
  - **"Weakened, not removed"** if instruct D has CI lower bound > 0, but the ΔD CI excludes 0.
  - Otherwise **"inconclusive"**.

## Part B — what an output norm would absorb (observational)
- Models: Qwen3-1.7B-Base, Qwen3-1.7B, Gemma-3-4B-it and Gemma-4-E4B-it. One **clean** forward pass per chunk; nothing is modified.
- At every unflagged layer and token, from the recorded `a`:
  - total write w = W_down a;
  - quiet write q = W_down(a·S);
  - mid write m = W_down(a·M), using the MID rule with seed 0.
- **Radial share** ρ(x) = (x·ŵ)² / ‖x‖², where ŵ = w/‖w‖. This is the fraction of a contribution that changes the *size* of the MLP output rather than its *direction*. An RMSNorm on w removes size changes to first order. For Gemma the gain vector is applied first (x → (1+γ)⊙x), since that is what the norm sees.
- Report the median ρ(q), the median ρ(m) and their ratio, per model.
- **P4B (norm-absorption hypothesis):** the median ρ(q) / median ρ(m) > 1.5 in **all four** models. The mechanism should be general, because it is meant to explain why a norm would hide the effect.
  - If it holds, quiet contributions are mainly size changes, which an output norm would absorb. That supports, but does not prove, the norm explanation.
  - If it fails, the norm explanation loses its simplest mechanism.

## Limitations
- One model family for Part A. Raw-text CE is not what instruct models are trained for.
- Part B is first-order and observational: it says what a norm *could* absorb, not what it causally does.
- Neurons, not SAE features.
