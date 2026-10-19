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
  - The effective rank is **~3–4 out of 128**, so each head's context-compiled local map is low-rank.
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

## Next steps
1. Pre-register and run a confirmatory Idea 2 test: new held-out texts and the energy-matched control. If it holds, repeat it with SAE features.
2. Idea 1: fit one shared vector field against layer-specific fits to the real residual updates, scored on held-out error (open per RELATED_WORK.md).
3. Idea 4 on a real model: measure the participation ratio of emotion representations across model sizes or checkpoints.
