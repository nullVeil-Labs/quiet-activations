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

> ### ⚠️ Correction (2026-10-20): the Gemma 3 results below are an artifact
> In Gemma-3-4B-it **layer 11**, the per-layer τ calibration gave τ₁₁ = 17.4, so *every* activation counted as "quiet", including the top-1% outliers. THRESH at that layer deleted the whole MLP, while the RAND/MID controls exclude outliers. A diagnostic found this in that one layer only (0/28 Qwen3 layers, 0/42 Gemma 4 layers).
> In an exploratory artifact check (`results/20261020-015709-ideas2a`, 40 fresh chunks), the quiet set was redefined to exclude outliers, as the controls do. The result:
> - **Gemma 3:** quiet − mid falls from **+0.239 [0.169, 0.317]** to **+0.019 [−0.007, 0.048]**.
> - **Qwen3:** **+0.034 → +0.032 [0.025, 0.040]**, essentially unchanged.
>
> **So the Gemma 3 "confirmation" (Phase 2 descriptive, Phase 3 PA1) and the "layer-11 hotspot" are not evidence for the hypothesis.** The effect stands in **Qwen3 only**. Gemma 4 is reversed, and Gemma 3 shows no clear effect. The sections below are kept as originally written for the record. The calibration rule was a design error in Phases 2–3. A future Phase 4 should use the outlier-excluded quiet definition from the start.

## Idea 2, Phase 3 — difference metric, layer profile, affect · PA1 ✓ PA2 ✓ PA3 ✗ PC1 ✗ · [real]
Run `results/20261019-201109-ideas2q`. Protocol `protocols/idea2_phase3.md`, frozen with its code at commit e6732c9. Part A uses 200 fresh chunks, Part B 20, Part C 480 template sentences. Per-chunk raw ΔCE is saved.

**A. Difference metric** (D = ΔCE quiet − ΔCE control, 95% bootstrap CI):

| | Qwen3-1.7B | Gemma-3-4B-it | Gemma-4-E4B-it |
|---|---|---|---|
| ΔCE quiet / mid / rand / noise | 0.061 / 0.029 / 0.029 / 0.029 | 0.217 / 0.009 / −0.011 / 0.068 | −0.012 / 0.029 / 0.025 / 0.005 |
| D vs MID | **+0.032** [0.028, 0.035] ✓ | **+0.208** [0.178, 0.241] ✓ | **−0.040** [−0.050, −0.030] ✗ |
| D vs RAND | +0.031 [0.028, 0.034] | +0.228 [0.200, 0.257] | −0.037 [−0.046, −0.027] |
| D vs NOISE | +0.031 [0.028, 0.035] | +0.150 [0.120, 0.180] | −0.016 [−0.027, −0.006] |

- **Qwen3:** quiet activations matter about 2× more than an equal energy of mid-tier activations. This is the third consistent result for Qwen3 (exploratory run, Phase 2, Phase 3).
- **Gemma 3:** confirmed and large. The controls do almost nothing, while quiet deletion costs 0.21 nats.
- **Gemma 4:** the pre-registered null failed, **in the opposite direction**. Quiet activations matter *less* than mid-tier ones, and deleting them slightly lowers CE.

**B. Layer profile (exploratory, 20 chunks, one layer ablated at a time):**
- **Gemma 3 — one hotspot, layer 11.** Quiet deletion there alone costs **0.327 nats**, more than all layers together (0.217). Every other layer is within ±0.035. At layer 11 it changes the layer output by 10.4%, about the whole MLP write (‖write‖/‖resid‖ = 0.11), against ~1% at other layers. Layer 11's MLP output seems to be carried mostly by quiet activations. Why is open.
- **Qwen3 — diffuse.** Single-layer quiet and mid deletion cost about the same (summed over layers: 0.061 vs 0.064). Yet with all layers ablated together, quiet costs 0.061 and mid only 0.029. Quiet-deletion damage roughly adds up across layers, while mid-tier damage partly does not. That is consistent with a quiet signal coordinated across layers. It is tentative: n = 20, and the A and B chunk sets differ.
- **Gemma 4 — no hotspot.** Single-layer effects are small and mixed in sign: slightly harmful in layers 0–12, slightly beneficial in later layers. Its MLP writes are sometimes much larger than the residual (‖write‖/‖resid‖ up to 13 at layer 23; 7.8 at layer 0). **This data does not explain why Gemma 4 differs.** Per-layer embeddings, KV sharing and the layer scalar all remain untested hypotheses.

**C. Does the quiet signal carry affect? Not supported (PC1 ✗ in both models).**

| | clean d′ valence / category | retention valence: quiet / mid | retention category: quiet / mid | interaction I [95% CI] |
|---|---|---|---|---|
| Qwen3 (layer 19) | 4.60 / 1.73 | 0.861 / 0.864 | 0.919 / 0.912 | 0.010 [−0.078, 0.089] |
| Gemma 3 (layer 23) | 2.11 / 1.53 | 0.928 / 1.002 | 0.983 / 0.924 | 0.134 [−0.019, 0.298] |
| Gemma 4 (layer 28) | 5.23 / 0.99 | 1.138 / 1.045 | 1.248 / 1.094 | 0.062 [−0.123, 0.251] |

- In Qwen3, quiet deletion erases valence and category information equally, with no sign of affect specificity.
- Gemma 3's point estimate leans toward the hypothesis, but the CI includes 0. The test may be underpowered, and that is not evidence for the hypothesis.

**Where Idea 2 stands:** some models (Qwen3 diffusely, Gemma 3 through one layer) depend on their many quiet MLP activations more than an equal energy elsewhere would predict. At least one model (Gemma 4) shows the reverse. We found no evidence that this signal is specifically affective.

## Idea 2 — layer-11 exploratory run and artifact check (2026-10-20) · [exploratory]
Plan: `protocols/idea2_layer11_exploratory.md`, committed before each run.
- **Layer-11 run** (`results/20261020-015453-ideas2l`): at layer 11, ‖q‖/‖w‖ = 1.000 and cos(q, w) = 1.000, and all 10,240 neurons were "quiet" on 100% of tokens. The "quiet write" *is* the whole MLP output there. Zeroing it costs 0.449 nats, and replacing it with its corpus mean costs −0.008, i.e. ≈ clean. In that layer, then, the MLP's contribution behaves mostly like a constant offset. That is a statement about layer 11's MLP, not about quiet activations. The token, direction and logit-lens readouts are saved but not interpreted, because they describe the whole MLP rather than a quiet tail.
- **Artifact check** (`results/20261020-015709-ideas2a`): see the correction notice above.

**Where Idea 2 stood after the correction:** in Qwen3-1.7B-Base, the many quiet MLP activations matter about 2× more than an equal energy of mid-tier activations. That held across four tests: run-1 follow-up, Phase 2, Phase 3, and the outlier-excluded check. It did not generalize to Gemma 3 (no clear effect) or Gemma 4 (reversed). There is no evidence that it is affect-specific.

## Idea 2, Phase 4 — training or architecture? · P4A1 ✓ · verdict "training removes it" · P4B ✗ · [real]
Run `results/20261020-023743-ideas2r`. Protocol `protocols/idea2_phase4.md`, frozen with its code at commit 4d86036.
- The quiet set uses the outlier-excluded definition.
- A guard skips layers where quiet covers more than 95% of non-outlier entries. That was layer 27 in both Qwen models (96%), and Gemma 3's layer 11 in Part B.
- 200 fresh chunks for Part A, 20 for Part B.
- The question came from the author's observation that the effect appeared only in a *base* model, while both Gemmas are *instruction-tuned*.

**A. Base vs instruct, same architecture (Qwen3-1.7B):**

| | Qwen3-1.7B-Base | Qwen3-1.7B (instruct) |
|---|---|---|
| clean CE | 2.876 | 3.431 |
| ΔCE quiet / mid | 0.065 / 0.025 | 0.050 / 0.042 |
| **D = quiet − mid** | **+0.040** [0.037, 0.044] | **+0.008** [−0.003, 0.023] |
| D / clean CE | 1.4% | 0.24% |
| chunks with D > 0 | 94% | 57% |

- **P4A1 ✓.** The base model replicates, the fifth consistent result for Qwen3-Base.
- **Instruct:** D is no longer distinguishable from 0. The paired difference is **ΔD = +0.032 [0.019, 0.043]**. By the pre-registered rule, the verdict is **"training removes it"**. More precisely, post-training reduced it by about 80%, to below detection. It is not shown to be exactly zero.
- The shift comes from both sides: quiet deletion hurts the instruct model a bit less (0.065 → 0.050), and mid-tier deletion hurts it clearly more (0.025 → 0.042). Post-training appears to move reliance *from* the quiet tail *toward* mid-tier activations.
- **Robustness checks** (post-run, not pre-registered):
  - The gap holds in each book separately. Base: D = 0.040 / 0.040. Instruct: 0.009 / 0.007.
  - Quiet and mid deletion are energy-matched in both models: mid removes 0.979 × the quiet energy in each.
- **Caveat:** CE on raw book text is not what instruct models are trained for. The relative D also falls (1.4% → 0.24%), so a worse baseline alone does not explain the drop, but the task mismatch remains.

**B. Would an output norm absorb quiet contributions? (observational, clean passes only) — P4B ✗**

| | post-FFN norm | median radial share ρ: quiet / mid | ratio |
|---|---|---|---|
| Qwen3-1.7B-Base | no | 0.034 / 0.026 | 1.32 |
| Qwen3-1.7B | no | 0.035 / 0.027 | 1.30 |
| Gemma-3-4B-it | yes | 0.066 / 0.057 | 1.15 |
| Gemma-4-E4B-it | yes | 0.088 / 0.077 | 1.14 |

- Only 3–9% of either kind of contribution lies along the MLP output's own direction. Both quiet and mid-tier contributions are overwhelmingly *directional*, and an RMSNorm cannot absorb directional change.
- The norm-absorption explanation (Gemini's hypothesis) loses its simplest mechanism. Gemma 4's early layers are an exception, with ratios of 2.8–5.0 in layers 0–9, which is worth a look but outside the prediction.

**Where Idea 2 stands now:**
- In Qwen3-1.7B, quiet MLP activations carry about 2× more weight than an equal energy of mid-tier activations **in the base model**, and **post-training largely removes this**, shifting reliance toward mid-tier activations.
- The two instruction-tuned Gemmas also show no positive effect, which is consistent with a training explanation. Without a Gemma *base* model, architecture is not ruled out for them.
- No evidence the signal is affect-specific. Neurons, not SAE features.

## Idea 2, Phase 5 — second family (SmolLM2-1.7B) · P5.1 ✓ · verdict "not training" · [real]
Run `results/20261020-035927-ideas2s`. Protocol `protocols/idea2_phase5.md`, frozen with its code at commit 575e7d8 and pushed before the run. The method is identical to Phase 4 Part A, on 200 fresh chunks. No layers were flagged by the guard (max 81% quiet).

| | SmolLM2-1.7B (base) | SmolLM2-1.7B-Instruct |
|---|---|---|
| clean CE | 2.242 | 2.513 |
| ΔCE quiet / mid | 0.141 / 0.120 | 0.114 / 0.093 |
| **D = quiet − mid** | **+0.022** [0.015, 0.029] | **+0.020** [0.014, 0.027] |
| D / clean CE | 0.97% | 0.81% |
| chunks with D > 0 | 73% | 69% |

- **P5.1 ✓.** The base-model effect replicates in a second, independent family (Llama architecture, different data and organisation). It is weaker than in Qwen3: quiet/mid ≈ 1.2× here vs ≈ 2.6× in Qwen3-Base.
- **The training verdict is "not training".** Paired ΔD = +0.001 [−0.006, 0.009]: post-training left the effect intact.
- **Robustness checks** (post-run): consistent across books (base 0.024 / 0.020; instruct 0.021 / 0.020), and energy-matched (mid/quiet energy = 0.99 in both).

**Where Idea 2 stands now:**
- Quiet MLP activations matter more than an equal energy of mid-tier ones in **both base models tested** (Qwen3-1.7B-Base, SmolLM2-1.7B) and in **SmolLM2-1.7B-Instruct**.
- They do not in **Qwen3-1.7B (instruct)**, **Gemma-3-4B-it** or **Gemma-4-E4B-it**.
- So post-training **can** remove the effect (Qwen3) but **does not always** (SmolLM2). Phase 4's "training removes it" is specific to Qwen3's post-training, not a general law.
- A hypothesis, not tested: the difference lies in the post-training recipe. Qwen3's pipeline is multi-stage and includes reinforcement learning and "thinking" training. SmolLM2-Instruct's is believed to be lighter (SFT + preference optimisation; not verified here).
- The Gemma nulls can't be attributed to training or architecture without a Gemma base model.

## Next steps
1. A Gemma **base** model (gated; needs a licence acceptance), to separate family from training for Gemma.
2. More base/instruct pairs whose post-training recipes are documented, to test the recipe hypothesis: is RL-heavy post-training what removes reliance on the quiet tail?
3. A dose-response check (scaling the quiet tail by 0–3×, with a mid-tier control) in base and instruct models.
4. *What* the quiet tail does where it matters, e.g. its effect on the calibration and entropy of next-token predictions. Then SAE features instead of neurons.
2. Idea 1: fit one shared vector field against layer-specific fits to the real residual updates, scored on held-out error (open per RELATED_WORK.md).
3. Idea 4 on a real model: measure the participation ratio of emotion representations across model sizes or checkpoints.
