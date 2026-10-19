# Analysis

Main run: `results/20261019-191428-ideas1234`. Exploratory follow-up: `results/20261019-191636-ideas2f`.
Model: Qwen3-1.7B-Base, on one RTX 4090. All samples are tiny (24 prompts or passages), and no significance tests were run. Read every number as descriptive.

Kinds of evidence:
- **[math]**: a proof or exact numerical identity.
- **[toy]**: a built model that shows what is possible.
- **[real]**: a measurement on Qwen3.
- **[exploratory]**: a test designed after seeing the data.

## Idea 1 — trajectories · pre-registered P1 ✓ P2 ✓ P3 ✓ · [real]
- **Turning:** the median cos(v_l, v_l+1) over layers 1–26 is **0.086**, so consecutive block writes are close to orthogonal. It is near 0 in early and middle layers and rises to 0.3–0.5 in layers 17–26, where the path gets straighter. **Caveat:** two unrelated vectors in 2048 dimensions also have cosine ≈ 0. "Curved" here means only that blocks write in different directions. It does **not** show a whirlpool or a smooth field.
- **Convergence:** R (within-group / between-group distance) falls from **1.12** (layer 0) to **0.71** (layer 25). It stays near 1 until about layer 17 and then drops, so prompts with the same answer do come together, but only in the last third of the network.
- **Wording comes first, meaning later:**
  - Same template, different answer (Paris vs Tokyo): the distance grows from 0 (an identical final token) to 1.10.
  - Same answer, different template: the distance shrinks from 1.40 to 0.86.
  - The two curves cross between layers 21 and 24. In the figure, Paris and Tokyo prompts travel *in pairs by wording* through the middle of the network.
- **Logit lens:** the answer is first top-1 at layers 18–22 in 5 of 6 groups. Monday is the exception, at layer 11.
- **The 2-D PCA explains only 20% of the variance.** The picture is a shadow of the real geometry, which is itself an argument against reading too much into single 2-D snapshots.
- **Not tested:** whether one shared field describes the motion. The closest prior evidence (see RELATED_WORK.md, Q3) suggests it does not.

## Idea 2 — sub-threshold "dark matter" · pre-registered P1 ✗ P2 ✗ P3 ✗ · [toy] + [real] + [exploratory]
- **Toy [toy]:** thresholding 50 aligned quiet features drops the tone readout from 3.10 to 0.47. This is true by construction.
- **How much is quiet:** at τ = 0.2, **67%** of MLP activations are below the threshold, but they hold only **4.5%** of the L2² energy. Zeroing all of them raises cross-entropy by **0.30 nats**, about 11% of the clean 2.81.
- **P1 (pre-registered) ✗:** random deletion "of equal mass" cost more (0.55–0.68). **But that control was not actually matched:** it removed 6.7–6.9% of the energy, because the frozen rule includes the crossing entry, and that entry is sometimes a huge outlier activation. The pre-registered verdict stands as ✗.
- **Exploratory follow-up [exploratory]:** with a control that removes ≤ the threshold's energy (4.3% vs 4.5%), random deletion costs **0.12–0.14** against **0.30** for threshold deletion. That is about **2.2× more harm per unit energy from deleting quiet activations**, the direction the hypothesis predicted. It needs a fresh pre-registered replication on new texts before it can be called a finding. Plausible reading: many small writes matter collectively, and the first control mostly measured the cost of hitting outlier neurons.
- **P2 ✗ (coherence):** the median C is **1.03** against **1.00** sign-shuffled. That is slightly above chance overall, but far from 1.2. The excess sits in layers 0–1 (1.44, 1.84), layer 3 (1.18) and the last layer (1.20). Middle layers are ≈ 1.0, the same as random vectors.
- **P3 ✗ (tone):** at τ = 0.2 the emotional/neutral ΔCE ratio is **1.06**. At τ = 0.1 it is 3.1, and at τ = 0.05 it reverses to 0.37. That pattern looks like noise from 12 passages per group, not a tone effect.
- **Caveat:** these are neurons, not SAE features.

## Idea 3 — KV-cache as weights · pre-registered P1 ✓ P2 ✓ P3 ✓ · [math] + [real]
- **Linear attention [math]:** o = W_fast·q holds to 1.8e-14. Recall is 1.0 in each context, and A's queries in context B score 0.0. Same weights, different function. rank(W_fast) equals the number of context pairs. This is the known fast-weights result (Schlag et al., 2021), shown here, not discovered.
- **Softmax Jacobian [math]:** the closed form matches finite differences to 4e-10.
- **Qwen3, layer 14 [real]:**
  - The per-head Jacobian ∂o/∂q differs between contexts by a median relative distance of **1.08**. Its magnitude is the same order as the Jacobian itself.
  - Repeating a context gives exactly 0.
  - The effective rank is **~3–4 out of 128**. **Correction (2026-10-19): this is largely expected, not a finding.** J = Vᵀ(diag(p) − ppᵀ)K/√d has rank < the number of attended tokens. Our contexts are ~15 tokens, and attention usually concentrates on a few of them (often an attention sink), which forces a low rank. It should not be read as a "context-steering subspace".
  - This is descriptive, from one layer.
- "Micro-self" and "relational engine" are metaphors. Nothing here measures selfhood.

## Idea 4 — empathy as compression · pre-registered P1 ✓ P2 ✓ P3 ✓ · [toy]
- **LATENT world:**
  - From N ≥ 1000, the 3-d latent model has the shortest code. At N = 20000, seed 0: 17.5k bits, against 1.17M for the rule table and 41.7k for the marginal.
  - Its held-out bits approach the true entropy: 0.818 vs 0.841. Being slightly under the entropy is test-sample noise.
  - **At N = 200 it overfits badly** (3.07 bits, worse than the marginal's 2.10), so compression needs data.
- **RULE world (control):** with no low-dimensional structure, the latent model is no better than the marginal (3.01 vs 3.00). The rule table is barely better (2.92). Nothing compresses an unstructured world.
- **Steering:** the latent model shifts every situation toward response 0 with **3** numbers, but the shift is uneven (mean 0.10 ± 0.10). The table needs **17,262** row edits (138k numbers), with a uniform shift.
- **What this does and does not say:** compact latent "state" models win **if and only if** the world has low-dimensional structure. The toy builds that structure in, so it cannot say whether human psychology has it. Choi & Weber (2026) find that LLM affect representations are fairly *diffuse*. Nothing here bears on whether a model *has* feelings.

## Idea 2, Phase 2 — confirmatory replication · P1 ✗ P2 ✗ P3 ✓ P4 ✓ · [real]
Run `results/20261019-192933-ideas2p`. Protocol `protocols/idea2_phase2.md`, frozen with its code at commit 50102ce before evaluation. 200 held-out Gutenberg chunks, 3 models.
**Verdict: the exploratory claim did not replicate as registered.** The effect turns out to be model-dependent.

τ_l is set per layer so the quiet set holds 5% of non-outlier energy, about 1–2% of total L2². Every control removes the same energy E. Ratios are ΔCE(THRESH)/ΔCE(control), with 95% paired-bootstrap CIs.

| | Qwen3-1.7B-Base | Gemma-3-4B-it | Gemma-4-E4B-it |
|---|---|---|---|
| clean CE | 2.904 | 3.499 | 4.491 |
| entries zeroed by THRESH | 72% | 62% | 65% |
| ΔCE THRESH (quiet) | **+0.062** | **+0.197** | −0.002 |
| ΔCE RAND (non-outlier, same E) | +0.031 | −0.016 | +0.023 |
| ΔCE MID (mid-tier, same E) | +0.030 | +0.004 | +0.027 |
| ΔCE MEAN (offset-preserving) | +0.043 | +0.467 | −0.057 |
| ΔCE NOISE (same output norm) | +0.029 | +0.071 | +0.007 |
| THRESH / RAND | **2.02** [1.89, 2.15] | −12.0 (denominator ≈ 0) | −0.07 [−0.68, 0.36] |
| THRESH / MID | **2.07** [1.93, 2.23] | 46.8 [−529, 449] (denominator ≈ 0) | −0.06 [−0.54, 0.32] |
| THRESH / NOISE | **2.17** [2.01, 2.34] | **2.76** [2.29, 3.44] | −0.24 [−3.3, 1.4] |
| MEAN / THRESH | 0.69 [0.65, 0.73] | 2.38 [2.11, 2.69] | 33.6 (denominator ≈ 0) |

- **P1 ✗ (direct replication, Qwen3, τ = 0.2):** THRESH / RAND = **1.39** [1.34, 1.43]. The direction replicates and the CI excludes 1, but the size misses the registered bar of 1.5. The exploratory 2.2× shrank on fresh text, as expected for an effect found after looking at the data.
- **P2 ✗ (quiet vs mid-tier):** strong only in Qwen3, so 1 of 3. In Gemma 3, quiet deletion clearly hurts (+0.20 nats) while every energy-matched control does essentially nothing. That makes the ratio meaningless, and the "strong" criterion can't pass. **A difference-based criterion would have been the better design; that is a flaw in the protocol.** Descriptively, Gemma 3 shows the largest quiet-specific damage of the three models. This reading is post hoc and labelled as such. In Gemma 4 there is no effect: deleting 65% of MLP entries (the quiet ones) changes CE by −0.002.
- **P3 ✓ (not just an offset):** in Qwen3, 69% of the damage survives mean-ablation. In Gemma 3, mean-ablation hurts *more* than zeroing. In both, a constant offset does not explain the harm.
- **P4 ✓ (direction matters):** in Qwen3 and Gemma 3, a random change of the same output size hurts 2.2–2.8× less. What hurts is the specific direction of the quiet sum, not just its size.
- **Interpretation:** in some models (Qwen3 clearly, Gemma 3 descriptively) the many quiet MLP activations carry functional, direction-specific signal beyond what their energy suggests. In Gemma 4 they appear to be genuinely ignorable. Why the models differ is open. Gemma-4-E4B has per-layer input embeddings and KV-sharing, which may route information differently. Untested.
- **Limitations:**
  - Per-chunk ΔCE values were not saved, only means and bootstrap results. This breaks the "preserve raw results" rule, and a rerun should save them.
  - The count confound remains.
  - Neurons are not SAE features.

## Next steps
1. Phase 3, if pursued: save per-chunk ΔCE, register a **difference**-based criterion (ΔCE THRESH − ΔCE control, with a CI), and explain the Gemma 4 null. If it holds, repeat with SAE features.
2. Idea 1: fit one shared vector field against layer-specific fits to the real residual updates, scored on held-out error (open per RELATED_WORK.md).
3. Idea 4 on a real model: measure the participation ratio of emotion representations across model sizes or checkpoints.
