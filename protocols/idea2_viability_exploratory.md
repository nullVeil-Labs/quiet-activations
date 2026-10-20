# Plan — Idea 2: is the quiet write structured, and is its job calibration? (EXPLORATORY)

Written 2026-10-20, committed together with its code before the run. **Exploratory**: the decision rules below are fixed in advance, but they are not confirmatory predictions.

## Question
Before hunting for "circuits" in the quiet tail, two viability checks:
1. **Structure.** At each layer, the quiet set is ~60–70% of MLP entries, and a neuron is quiet on some tokens and loud on others. So "quiet neurons" are not fixed units. The object our phases showed to matter is the **quiet write** q_l = W_out (a ⊙ S) at each token. Is q low-dimensional? If a few directions per layer carry the damage, there is structure to trace. If the damage is spread over hundreds of directions, neuron- or direction-level circuit hunting is unlikely to work, and that is the finding.
2. **Calibration.** A candidate job that would explain the base-vs-instruct split: base models need fine adjustments to spread probability over plausible next tokens; instruct tuning sharpens outputs and stops using them. If so, much of the quiet-specific damage should be undone by re-tuning one softmax temperature.

## Setup
- **Models:** Qwen3-1.7B-Base, Qwen3-1.7B, SmolLM2-1.7B, SmolLM2-1.7B-Instruct, Gemma-3-1B-pt (same snapshots as Phases 4–6).
- **New corpus** (not used by any earlier run): Project Gutenberg #84 (*Frankenstein*) and #2701 (*Moby-Dick*), `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt`, saved as `data/pg84.txt` and `data/pg2701.txt`.
  - sha256 pg84.txt `7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3`
  - sha256 pg2701.txt `b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d`
  - Same chunking as Phase 2 (334 and 947 chunks), shuffled with `random.Random(0)`. Per book: [0:20] τ calibration, [20:50] direction fitting, [50:100] evaluation (100 chunks). **[100:] is reserved** for later confirmatory tests and is not read here.
- **Quiet set, τ, guard:** as Phases 4–6: outlier-excluded quiet set, τ_l = 5% of non-outlier energy, layers with > 95% quiet entries never touched. Per model.
- **Directions:** on the direction-fitting chunks (clean passes), per layer: the mean quiet write q̄_l and the eigenvectors V_l of the centred covariance of q_l, sorted by variance. A random orthonormal basis R_l (seed 0) is the control.

## Conditions (all non-guarded layers at once; ΔCE vs clean, per chunk)
- **ZERO:** q removed (Phase 4 THRESH).
- **MID:** energy-matched mid-tier write removed (Phase 4 rule), seeds 0, 1, 2, averaged.
- **MEAN:** q → q̄_l.
- **PC_k:** q → q̄_l + P_{V_l,k}(q − q̄_l), keep the top-k directions, k ∈ {1, 4, 16, 64, 256}.
- **RAND_k:** the same with k random directions, k ∈ {16, 64, 256}.

## Measures
- ΔCE, Δ mean entropy, Δ top-1 accuracy, ECE (15 bins, pooled tokens), mean KL(clean ‖ condition).
- **D = ZERO − MID** per chunk (the quiet-specific effect, as in Phases 4–6), with a bootstrap CI. This also replicates the main effect on new text.
- **Recovered fraction** f(c) = 1 − ΔCE_c / ΔCE_ZERO.
- **Dimensionality:** per layer, the participation ratio of the centred eigenvalues, the share of E‖q‖² in q̄, and the centred variance in the top k.
- **Temperature test:** CE at T ∈ {0.60, 0.65, …, 1.60}. T is fitted per condition on the even-indexed eval chunks, then scored on the odd ones. D_T = (ZERO at its T*) − (MID at its T*). Temperature-recovered share = 1 − D_T / D (odd chunks).

## Decision rules (fixed now; judged only for models whose D has a 95% CI above 0 on this corpus)
- **Structure:** "low-dimensional" if f(PC_16) − f(RAND_16) ≥ 0.30. "Diffuse" if f(PC_256) − f(RAND_256) < 0.10. Otherwise "intermediate".
- **Calibration:** "calibration-like" if the temperature-recovered share of D ≥ 0.50. "Not calibration-like" if < 0.20. Otherwise "partial".
- **Descriptive only:** clean entropy and ECE per model. If the calibration story is right, SmolLM2-Instruct (the instruct model that kept the effect) should look more base-like than Qwen3-1.7B. With 5 models this is a hint, not a test.

## Limitations
- One temperature is a crude stand-in for "calibration"; a "not calibration-like" result does not rule out subtler distribution-shape roles.
- Directions are fitted on clean activations; under multi-layer ablation, upstream changes shift q, so PC_k is not a pure projection of the clean write.
- PCA finds directions of variance, not of causal importance; a low-variance direction could matter more. "Diffuse" here means diffuse in the variance basis.
- Exploratory, one seed for the random basis, no multiple-comparison correction. Anything promising needs a confirmatory run on the reserved chunks.

## Note before the run
The code was smoke-tested on SmolLM2-1.7B-Instruct with 6 calibration chunks and 8 direction-fitting chunks (4 fitting, 4 standing in for evaluation). No evaluation chunk was read. Those numbers were seen before this commit; the decision rules above were written before the smoke test and are unchanged.
