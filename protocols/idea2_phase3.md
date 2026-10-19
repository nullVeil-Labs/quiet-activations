# Protocol — Idea 2, Phase 3: better metric, where it lives, and what it carries

Frozen: 2026-10-19, committed with the exact code (`nightshift/phase3.py`) before any Phase 3 evaluation data was run.
It follows Phase 2 (`idea2_phase2.md`, run 20261019-192933), which did not replicate as registered and showed a model-dependent effect.

## Shared setup (unchanged from Phase 2)
- The same three models, BOS handling, MLP `down_proj` input site, per-layer τ_l (5% of non-outlier L2² energy), outlier definition (top 1% per position), and the THRESH / RAND / MID / NOISE conditions with seeds 0–2.
- τ_l and the quiet means are recalibrated on the same 40 calibration chunks as Phase 2.
- The same Gutenberg books, the same seed-0 shuffle. Per book, Phase 2 used chunks [0:20] (calibration) and [20:120] (evaluation). Phase 3 uses **[120:220] for Part A** (200 chunks) and **[220:230] for Part B** (20 chunks). No Phase 3 chunk was used by any earlier run.

## Part A — difference metric (confirmatory)
- Per chunk: D_X = ΔCE(THRESH) − ΔCE(X), for X ∈ {RAND, MID, NOISE}. Seeded conditions are averaged over seeds per chunk. Mean D with a 95% bootstrap CI (2000 resamples over chunks, seed 0).
- **Per-chunk ΔCE for every condition and seed is saved** (this fixes Phase 2's missing raw data).
- **PA1:** Gemma-3-4B-it: mean D_MID > 0.01 nats and CI lower bound > 0.
- **PA2:** Qwen3-1.7B: the same.
- **PA3:** Gemma-4-E4B-it: the CI of D_MID includes 0 (a pre-registered null, based on Phase 2).

## Part B — per-layer profile (exploratory; no predictions)
On the 20 Part-B chunks, for each layer l separately (only layer l ablated):
- ΔCE for THRESH, MID (seed 0) and KO (the whole MLP output zeroed).
- The relative change of the layer's residual write (`post_feedforward_layernorm` output for Gemma, `down_proj` output for Qwen) and of the layer output.
- The baseline ratio ‖write‖ / ‖residual‖.

This describes where quiet activations matter and how much of their change survives normalization. Any explanation of the Gemma 4 null drawn from it is a hypothesis.

## Part C — does the quiet signal carry affect? (confirmatory)
- 480 template sentences in a 2 × 2 factorial: **valence** (8 positive vs 8 negative emotion words) × **category** (4 vehicles vs 4 furniture words), 120 per cell. Frames, names and word lists are fixed in the code. Train/test split 240/240, stratified, with no repeated sentence.
- Representation: the residual stream at the final token, at layer round(2L/3).
- Directions: the difference of class means on clean *train* representations, fixed afterwards.
- d′ on test representations: clean, THRESH (all layers ablated), and MID (seeds 0–2, d′ averaged).
- Retention = d′(ablated) / d′(clean), computed separately for valence and category.
- **Interaction I** = [ret_cat − ret_val]_THRESH − [ret_cat − ret_val]_MID. I > 0 means deleting quiet activations erases valence information more than category information, beyond what deleting the same energy from mid-tier activations does. 95% CI from 2000 bootstrap resamples of test sentences (directions fixed).
- **PC1:** I > 0.01 and CI lower bound > 0, in **both** Qwen3-1.7B and Gemma-3-4B-it.
- Gemma 4: reported, no prediction.

## Disclosure
Before freezing, the code was smoke-tested on calibration chunks only, with placeholder labels for Part C (not the real sentences). Those numbers are not results. No bugs were found.

## Limitations
- Template sentences are artificial.
- Valence and category words differ in frequency and in position within the sentence.
- The probe is a linear difference of means at one layer.
- Neurons, not SAE features.
- The books are probably in the training data.
