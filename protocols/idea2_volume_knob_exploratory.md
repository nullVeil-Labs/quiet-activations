# Plan — Idea 2: the quiet-volume knob (EXPLORATORY)

Written 2026-10-20, committed before the run. **Exploratory**: the expectations below are stated in advance, but they are not confirmatory predictions.

## Question
What happens when the quiet tail is scaled **up** as well as down? And does turning up the quiet tail of an instruct model make its next-token predictions **more like its base model's**?

## Setup
- **Pairs:**
  - Qwen3-1.7B-Base / Qwen3-1.7B: post-training removed the effect.
  - SmolLM2-1.7B / SmolLM2-1.7B-Instruct: post-training did not.
- **Site, quiet set, calibration:** the same as Phases 4–6: outlier-excluded quiet set, per-layer τ_l (5% of non-outlier energy) on the same 40 calibration chunks, > 95% guard. τ_l is calibrated **per model**.
- **Data:** the last unused Gutenberg chunks, [560:] of each shuffled book: 47 + 11 = 58 chunks. This is the only fresh text left. Another confirmatory test would need a new corpus.
- **Interventions, all layers:**
  - **QUIET×k:** a_S ← k·a_S.
  - **MID×k:** a_M ← k·a_M, where M is the energy-matched mid-tier set (Phase 4 rule, seed 0).
  - k ∈ {0, 0.5, 1, 1.5, 2, 3}. k = 1 is clean.

## Measures (per model, per condition)
1. **ΔCE** vs clean, on the chunk text.
2. **Instruct models only:** the mean per-token KL(p_base ‖ p_instruct,k), where p_base is the *clean* base model on the same text. This asks: does the intervention move the instruct model toward its base?

## Expectations (stated in advance)
- **E1:** in base models, ΔCE is minimal at k = 1 for both QUIET and MID (an inverted-U in quality). The model is tuned to its own activations.
- **E2:** in Qwen3-1.7B (instruct), QUIET×k for k > 1 lowers KL-to-base more than MID×k does. Its post-training reduced reliance on the quiet tail, so turning it up might partly undo that.
- **E3:** in SmolLM2-Instruct, there is no QUIET-specific KL reduction, since its quiet-tail use is unchanged from base.
- If E2 fails, turning up the quiet tail does not restore base-like behaviour. That would mean post-training changed *how* the quiet signal is used, not merely its weight.

## Limitations
- 58 chunks, a single MID seed, exploratory only.
- KL to base measures base-likeness, not quality or creativity.
