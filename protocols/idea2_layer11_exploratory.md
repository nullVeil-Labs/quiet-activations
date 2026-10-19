# Plan — Idea 2: what is in Gemma-3-4B-it layer 11? (EXPLORATORY)

Written 2026-10-20, committed before the run. **Exploratory, no confirmatory predictions.** Anything striking needs its own frozen protocol and fresh data.

## Why
In Phase 3 Part B (run 20261019-201109), deleting quiet MLP activations at layer 11 alone cost 0.327 nats, more than deleting them at all layers (0.217). Every other layer was within ±0.035.

## Setup
- Gemma-3-4B-it, same site, τ calibration and BOS handling as Phase 2/3 (`nightshift/phase2.py`).
- Data: Gutenberg chunks **[230:250] per book** (40 chunks). No earlier run used them.
- Ablations at layer 11 only.

## Measures
1. **Why this layer (all layers, clean):** with total MLP write w = W_down a, split into a quiet part q = W_down(a·S) and a loud part w − q. Report ‖q‖/‖w‖, cos(q, w) and ‖w‖/(‖q‖ + ‖w − q‖) (1 = no cancellation, ≪ 1 = loud parts cancel). Median over tokens, per layer.
2. **Which tokens:** per-token ΔCE for quiet deletion at layer 11. Report ΔCE by position (first 8 positions vs the rest) and the tokens with the largest mean ΔCE (≥ 5 occurrences).
3. **One direction or many:** PCA of q over all tokens. Report variance in PC1–3, and the mean cosine of q to its mean direction. The same for the mid-tier write of equal energy, as a comparison.
4. **Bias or information:**
   - **MEANQ:** replace q at each token with the corpus-mean quiet write q̄ (from calibration chunks).
   - **ZEROQ:** zero q (= THRESH at layer 11).
   - **PCQ:** keep only q's projection onto its top-k PCs (k = 1, 3, 10).
   Report ΔCE for each. If MEANQ ≈ clean, the effect is a constant offset.
5. **Which neurons:** how often each neuron is quiet. Count neurons that are quiet ≥ 99% of tokens, and their share of ‖q‖².
6. **What it says:** a logit lens on q̄ through the final norm and unembedding, top-15 promoted and suppressed tokens. A direct read, so treat it as suggestive only.

## Addendum — 2026-10-20, written after the layer-11 run (exploratory)
The layer-11 run found τ_11 = 17.4 for Gemma 3: **every** activation at layer 11 counted as quiet, including the top-1% outliers. So THRESH at that layer deleted the whole MLP, while the RAND/MID controls exclude outliers. A diagnostic on calibration chunks found this in Gemma-3 layer 11 only: 0/28 Qwen3 layers and 0/42 Gemma-4 layers are affected. Position 0 is not the cause (its max |a| at L11 is 0.9).
Artifact check (`python run.py 2a`): on the same 40 layer-11 chunks, for Qwen3 and Gemma 3, compare all-layer THRESH with all-layer MID (seeds 0–2) under two quiet definitions:
- (i) the original |a| < τ_l;
- (ii) **|a| < τ_l and not an outlier**, the same exclusion the controls use. E is recomputed from (ii).

If Gemma 3's quiet-minus-mid difference disappears under (ii), its Phase 2/3 result was an artifact of layer 11. Exploratory: a definitive answer needs a frozen Phase 4.
