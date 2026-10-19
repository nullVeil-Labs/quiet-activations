# Protocol — Idea 1: reasoning as trajectories

Frozen: 2026-10-19, before any run.

## Question
Does the last-token residual stream move through layers along a curved path, and do prompts with the same answer converge (an "attractor") as depth increases?

## Setup
- Model: Qwen3-1.7B base, bf16 weights, fp32 analysis. 28 layers, so states h_0 (embeddings) … h_28.
- Prompts: 6 answer groups × 4 paraphrases = 24 prompts, fixed in `nightshift/trajectories.py` (`PROMPTS`).
- State: hidden state of the final prompt token at every layer.

## Measures
- **M1 turning.** Velocity v_l = h_{l+1} − h_l (the block's write to the residual stream). Turning = cos(v_l, v_{l+1}). Report the median over prompts per layer pair.
- **M2 convergence.** Per layer: subtract the mean state over all 24 prompts, then unit-normalize. R_l = mean within-group distance / mean between-group distance.
- **M3 logit lens.** Final norm + unembedding applied to each h_l. Report the first layer where the group's answer token is top-1.
- Figure: PCA (fit on all centered, normalized states from layers 1–28) with each prompt's path drawn in it, plus a single-layer snapshot panel for comparison.

## Predictions (stated before running)
- P1 (curved): median turning cosine over layers 1–26 is < 0.5. If it's ≥ 0.9, the paths are close to straight and P1 fails.
- P2 (attractor): min over l ≥ 1 of R_l < 0.7 × R_1.
- P3: the answer becomes top-1 under the logit lens only in the second half of the network (layer > 14) for most groups.

## Caveats (written in advance)
- 24 prompts is a tiny descriptive sample, with no significance testing.
- Turning between consecutive block outputs could come from the architecture (blocks write into different subspaces) rather than "reasoning". A curved path is necessary for the fluid picture, but it doesn't confirm it.
- The PCA is a projection. Distances in it aren't distances in 2048-d space.
