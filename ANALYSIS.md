# Analysis

**Run 1 (Ideas 1–4):** `results/20261019-191428-ideas1234`; exploratory follow-up `results/20261019-191636-ideas2f`. Qwen3-1.7B-Base on one RTX 4090, tiny samples (24 prompts or passages), no significance tests, so read run-1 numbers as descriptive. For run 1 and 2f, only the protocols were frozen before running; the code was committed together with the results. From Phase 2 on, each protocol was frozen together with its code. Later sections use more models and bootstrap CIs.

**Fact-check (2026-10-20):** six independent read-only reviews checked every number against the raw JSON and audited the git history. No protocol's frozen text was edited after its run, and no result preceded its protocol. The wording, scope and precision fixes they found are applied below.

Kinds of evidence:
- **[math]**: a proof or exact numerical identity.
- **[toy]**: a built model that shows what is possible.
- **[real]**: a measurement on a real model.
- **[exploratory]**: a test designed after seeing the data.

## Idea 1 — trajectories · pre-registered P1 ✓ P2 ✓ P3 ✓ · [real]
- **Turning:** the median cos(v_l, v_l+1) over layers 1–26 is **0.086**, so consecutive block writes are close to orthogonal. It is near 0 in early and middle layers and rises to 0.25–0.5 in layers 17–26, where the path gets straighter (the last pair drops to −0.18). **Caveat:** two unrelated vectors in 2048 dimensions also have cosine ≈ 0. "Curved" here means only that blocks write in different directions. It does **not** show a whirlpool or a smooth field.
- **Convergence:** R (within-group / between-group distance) falls from **1.12** (layer 0) to **0.71** (layer 25). It stays near 1 until about layer 17 and then drops, so prompts with the same answer do come together, but only in the last third of the network.
- **Wording comes first, meaning later:**
  - Same template, different answer (Paris vs Tokyo): the distance grows from 0 (an identical final token) to 1.10.
  - Same answer, different template: the distance shrinks from ~1.4–1.5 to a minimum of 0.85 (layer 25).
  - The two curves cross between layers 23 and 24. In the figure, Paris and Tokyo prompts travel *in pairs by wording* through the middle of the network.
- **Logit lens:** the answer is first top-1 at layers 18–22 in 5 of 6 groups. Monday is the exception, at layer 11.
- **The 2-D PCA explains only 20% of the variance.** The picture is a shadow of the real geometry, which is itself an argument against reading too much into single 2-D snapshots.
- **Not tested:** whether one shared field describes the motion. The closest prior evidence (see RELATED_WORK.md, Q3) suggests it does not.

## Idea 2 — sub-threshold "dark matter" · pre-registered P1 ✗ P2 ✗ P3 ✗ · [toy] + [real] + [exploratory]
- **Toy [toy]:** thresholding 50 aligned quiet features drops the tone readout from 3.10 to 0.47. This is true by construction.
- **How much is quiet:** at τ = 0.2, **67%** of MLP activations are below the threshold, but they hold only **4.5%** of the L2² energy. Zeroing all of them raises cross-entropy by **0.30 nats**, about 11% of the clean 2.81.
- **P1 (pre-registered) ✗:** random deletion "of equal mass" cost more (0.55–0.68). **But that control was not actually matched:** it removed 6.7–6.9% of the energy, because the frozen rule includes the crossing entry, and that entry is sometimes a huge outlier activation. The pre-registered verdict stands as ✗.
- **Exploratory follow-up [exploratory]:** with a control that removes ≤ the threshold's energy (4.3% vs 4.5%), random deletion costs **0.12–0.14** against **0.30** for threshold deletion. That is about **2.2× more harm per unit energy from deleting quiet activations**, the direction the hypothesis predicted. It needs a fresh pre-registered replication on new texts before it can be called a finding. Plausible reading: many small writes matter collectively, and the first control mostly measured the cost of hitting outlier neurons.
- **P2 ✗ (coherence):** the median C is **1.03** against **1.00** sign-shuffled. That is slightly above chance overall, but far from 1.2. The excess sits in layers 0–1 (1.44, 1.84), layer 3 (1.18) and the last layer (1.20). Layers 4–10 sit slightly above chance (1.06–1.10); layers 11–26 are ≈ 1.01, the same as random vectors.
- **P3 ✗ (tone):** at τ = 0.2 the emotional/neutral ΔCE ratio is **1.06**. At τ = 0.1 it is 3.1, and at τ = 0.05 it reverses to 0.37. That pattern looks like noise from 12 passages per group, not a tone effect.
- **Caveat:** these are neurons, not SAE features.

## Idea 3 — KV-cache as weights · pre-registered P1 ✓ P2 ✓ P3 ✓ · [math] + [real]
- **Linear attention [math]:** o = W_fast·q holds to 1.8e-14. Recall is 1.0 in each context, and A's queries in context B score 0.0. Same weights, different function. rank(W_fast) equals the number of context pairs (tested up to d = 32). This is the known fast-weights result (Schlag et al., 2021), shown here, not discovered.
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
  - **At N = 200 it overfits badly** (3.07 bits, worse than the marginal's 2.10), so compression needs data. At N = 200 its two-part code is still the shortest in 2 of 3 seeds even though it generalizes worst, so code length alone did not flag the overfit.
- **RULE world (control):** with no low-dimensional structure, the latent model is no better than the marginal (3.01 vs 3.00). The rule table is barely better (2.92). Nothing compresses an unstructured world.
- **Steering:** in the LATENT world (N = 20000, seed 0), the latent model shifts every situation toward response 0 with **3** numbers, but unevenly (mean 0.10 ± 0.10; the std is 0.09–0.14 across seeds). In the RULE world the same 3-number edit moves p0 the *wrong* way (−0.09 to −0.13, 0% of situations up). The table needs **17,261** row edits + 1 fallback (138k numbers), with a uniform shift.
- **What this does and does not say:** the compact latent model wins in the world built with low-dimensional structure and does not win in the unstructured control. Two built worlds cannot show "if and only if". The toy builds that structure in, so it cannot say whether human psychology has it. Choi & Weber (2026) find that LLM affect representations are fairly *diffuse*. Nothing here bears on whether a model *has* feelings.

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
- **P3 ✓ (not just an offset):** in Qwen3, 69% of the damage survives mean-ablation. In Gemma 3, mean-ablation hurts *more* than zeroing. In both, a constant offset does not explain the harm. ⚠️ The Gemma 3 part is affected by the layer-11 artifact (see the correction below); at that layer, replacing the MLP write with its mean is ≈ clean.
- **P4 ✓ (direction matters):** in Qwen3 and Gemma 3, a random change of the same output size hurts 2.2–2.8× less. What hurts is the specific direction of the quiet sum, not just its size. ⚠️ Without the artifact-affected Gemma 3, P4 holds in 1 of 3 models.
- **Interpretation:** in some models (Qwen3 clearly, Gemma 3 descriptively) the many quiet MLP activations carry functional, direction-specific signal beyond what their energy suggests. In Gemma 4 they appear to be genuinely ignorable. ⚠️ The Gemma 3 part of this reading was later retracted (see the correction). Why the models differ is open. Gemma-4-E4B has per-layer input embeddings and KV-sharing, which may route information differently. Untested.
- **Limitations:**
  - Per-chunk ΔCE values were not saved, only means and bootstrap results. This breaks the "preserve raw results" rule, and a rerun should save them.
  - The count confound remains.
  - Neurons are not SAE features.

> ### ⚠️ Correction (2026-10-20): the Gemma 3 results in Phase 2 (above) and Phase 3 (below) are an artifact
> In Gemma-3-4B-it **layer 11**, the per-layer τ calibration gave τ₁₁ = 17.4, so *every* activation counted as "quiet", including the top-1% outliers. THRESH at that layer deleted the whole MLP, while the RAND/MID controls exclude outliers. A diagnostic found this in that one layer only (0/28 Qwen3 layers, 0/42 Gemma 4 layers; a diagnostic on calibration text, recorded in the layer-11 plan addendum, not saved as a results file).
> In an exploratory artifact check (`results/20261020-015709-ideas2a`, 40 fresh chunks), the quiet set was redefined to exclude outliers, as the controls do. The result:
> - **Gemma 3:** quiet − mid falls from **+0.239 [0.169, 0.317]** to **+0.019 [−0.007, 0.048]** (n = 40, exploratory). Layer 11 stays uncontrolled even under (ii): every non-outlier entry there is still quiet, so MID removes nothing at that layer.
> - **Qwen3:** **+0.034 → +0.032 [0.025, 0.040]**, essentially unchanged.
>
> **So the Gemma 3 "confirmation" (Phase 2 descriptive, Phase 3 PA1) and the "layer-11 hotspot" are not evidence for the hypothesis.** At that point the effect stood in **Qwen3 only** (Phases 5–6 later added SmolLM2 and Gemma-3-1B-pt). Gemma 4 is reversed, and Gemma-3-4B-it shows no clear effect. **Phase 2's P3 ✓ and P4 ✓ also rely on Gemma 3:** without it, P4 holds in 1 of 3 models and P3 only through Gemma 4's undefined ratio, so treat both as unsupported. The sections below are kept as originally written for the record. The calibration rule was a design error in Phases 2–3. A future Phase 4 should use the outlier-excluded quiet definition from the start.

## Idea 2, Phase 3 — difference metric, layer profile, affect · PA1 ✓ (⚠️ artifact, see correction) PA2 ✓ PA3 ✗ PC1 ✗ · [real]
Run `results/20261019-201109-ideas2q`. Protocol `protocols/idea2_phase3.md`, frozen with its code at commit e6732c9. Part A uses 200 fresh chunks, Part B 20, Part C 480 template sentences. Per-chunk raw ΔCE is saved.

**A. Difference metric** (D = ΔCE quiet − ΔCE control, 95% bootstrap CI):

| | Qwen3-1.7B | Gemma-3-4B-it | Gemma-4-E4B-it |
|---|---|---|---|
| ΔCE quiet / mid / rand / noise | 0.061 / 0.029 / 0.029 / 0.029 | 0.217 / 0.009 / −0.011 / 0.068 | −0.012 / 0.028 / 0.025 / 0.005 |
| D vs MID | **+0.032** [0.028, 0.035] ✓ | **+0.208** [0.178, 0.241] ✓ | **−0.040** [−0.050, −0.030] ✗ |
| D vs RAND | +0.031 [0.028, 0.034] | +0.228 [0.200, 0.257] | −0.037 [−0.046, −0.027] |
| D vs NOISE | +0.031 [0.028, 0.035] | +0.150 [0.120, 0.180] | −0.016 [−0.027, −0.006] |

- **Qwen3:** quiet activations matter about 2× more than an equal energy of mid-tier activations. This is the third consistent result for Qwen3 (exploratory run, Phase 2, Phase 3).
- **Gemma 3:** [⚠️ retracted: calibration artifact, see correction] confirmed and large. The controls do almost nothing, while quiet deletion costs 0.21 nats.
- **Gemma 4:** the pre-registered null failed, **in the opposite direction**. Quiet activations matter *less* than mid-tier ones, and deleting them slightly lowers CE.

**B. Layer profile (exploratory, 20 chunks, one layer ablated at a time):**
- **Gemma 3 — one hotspot, layer 11.** [⚠️ retracted: this "hotspot" is the calibration artifact] Quiet deletion there alone costs **0.327 nats**, more than all layers together (0.217, on the Part A chunks). Every other layer is within ±0.035. At layer 11 it changes the layer output by 10.4%, about the whole MLP write (‖write‖/‖resid‖ = 0.11), against 1–5% in layers 0–10 and ~1% in layers 12–33. Layer 11's MLP output seems to be carried mostly by quiet activations. Why is open.
- **Qwen3 — diffuse.** Single-layer quiet and mid deletion cost about the same (summed over layers: 0.061 vs 0.064). Yet with all layers ablated together, quiet costs 0.061 and mid only 0.029. Quiet-deletion damage roughly adds up across layers, while mid-tier damage partly does not. That is consistent with a quiet signal coordinated across layers. It is tentative: n = 20, and the A and B chunk sets differ.
- **Gemma 4 — no hotspot.** Single-layer effects are small and mixed in sign: slightly harmful in layers 0–12, slightly beneficial in later layers. Its MLP writes are sometimes much larger than the residual (‖write‖/‖resid‖ up to 13 at layer 23; 7.8 at layer 0). **This data does not explain why Gemma 4 differs.** Per-layer embeddings, KV sharing and the layer scalar all remain untested hypotheses.

**C. Does the quiet signal carry affect? Not supported (PC1 ✗ in both models).**

| | clean d′ valence / category | retention valence: quiet / mid | retention category: quiet / mid | interaction I [95% CI] |
|---|---|---|---|---|
| Qwen3 (layer 19) | 4.60 / 1.73 | 0.861 / 0.864 | 0.919 / 0.912 | 0.010 [−0.078, 0.089] |
| Gemma 3 (layer 23) | 2.11 / 1.53 | 0.928 / 1.002 | 0.983 / 0.924 | 0.134 [−0.019, 0.298] |
| Gemma 4 (layer 28) | 5.23 / 0.99 | 1.138 / 1.045 | 1.248 / 1.094 | 0.062 [−0.123, 0.251] |

- In Qwen3, quiet deletion erases valence and category information no differently than mid-tier deletion does (I = 0.010 [−0.078, 0.089]), with no sign of affect specificity.
- Gemma 3's point estimate leans toward the hypothesis, but the CI includes 0. The test may be underpowered, and that is not evidence for the hypothesis.

**Where Idea 2 stood before the correction (superseded):** some models (Qwen3 diffusely, Gemma 3 through one layer) depend on their many quiet MLP activations more than an equal energy elsewhere would predict. At least one model (Gemma 4) shows the reverse. We found no evidence that this signal is specifically affective.

## Idea 2 — layer-11 exploratory run and artifact check (2026-10-20) · [exploratory]
Plan: `protocols/idea2_layer11_exploratory.md`, committed before each run.
- **Layer-11 run** (`results/20261020-015453-ideas2l`): at layer 11, ‖q‖/‖w‖ = 1.000 and cos(q, w) = 1.000, and all 10,240 neurons were "quiet" on ≥ 99% of tokens (median 100%). The "quiet write" *is* the whole MLP output there. Zeroing it costs 0.449 nats, and replacing it with its corpus mean costs −0.008, i.e. ≈ clean. In that layer, then, the MLP's contribution behaves mostly like a constant offset. That is a statement about layer 11's MLP, not about quiet activations. The token, direction and logit-lens readouts are saved but not interpreted, because they describe the whole MLP rather than a quiet tail.
- **Artifact check** (`results/20261020-015709-ideas2a`): see the correction notice above.

**Where Idea 2 stood after the correction:** in Qwen3-1.7B-Base, the many quiet MLP activations matter about 2× more than an equal energy of mid-tier activations (Phase 2, Phase 3, outlier-excluded check) or random activations (run-1 follow-up). The direction held in all four; Phase 2's registered fixed-τ replication gave a smaller 1.39× and missed its size bar. It did not generalize to Gemma 3 (no clear effect) or Gemma 4 (reversed). There is no evidence that it is affect-specific.

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
| ΔCE quiet / mid | 0.065 / 0.024 | 0.050 / 0.041 |
| **D = quiet − mid** | **+0.040** [0.037, 0.044] | **+0.008** [−0.003, 0.023] |
| D / clean CE | 1.4% | 0.24% |
| chunks with D > 0 | 94% | 57% |

- **P4A1 ✓.** The base model replicates, the fifth consistent result for Qwen3-Base.
- **Instruct:** D is no longer distinguishable from 0. The paired difference is **ΔD = +0.032 [0.019, 0.043]**. By the pre-registered rule, the verdict is **"training removes it"**. More precisely, post-training reduced it by about 80%, to below detection. It is not shown to be exactly zero.
- The shift comes from both sides: quiet deletion hurts the instruct model a bit less (0.065 → 0.050), and mid-tier deletion hurts it clearly more (0.024 → 0.041). Post-training appears to move reliance *from* the quiet tail *toward* mid-tier activations.
- **Robustness checks** (post-run, not pre-registered):
  - The gap holds in each book separately. Base: D = 0.040 / 0.040. Instruct: 0.009 / 0.007.
  - Quiet and mid deletion are energy-matched in both models: mid removes 0.979 × the quiet energy in each (from the post-run check log; not saved in the results JSON).
- **Caveat:** CE on raw book text is not what instruct models are trained for. The relative D also falls (1.4% → 0.24%), so a worse baseline alone does not explain the drop, but the task mismatch remains.

**B. Would an output norm absorb quiet contributions? (observational, clean passes only) — P4B ✗**

| | post-FFN norm | median radial share ρ: quiet / mid | ratio |
|---|---|---|---|
| Qwen3-1.7B-Base | no | 0.034 / 0.026 | 1.32 |
| Qwen3-1.7B | no | 0.035 / 0.027 | 1.30 |
| Gemma-3-4B-it | yes | 0.066 / 0.057 | 1.15 |
| Gemma-4-E4B-it | yes | 0.088 / 0.077 | 1.14 |

- Only 3–9% of either kind of contribution lies along the MLP output's own direction. Both quiet and mid-tier contributions are overwhelmingly *directional*, and an RMSNorm cannot absorb directional change.
- The norm-absorption explanation (Gemini's hypothesis) loses its simplest mechanism. Gemma 4's early layers are an exception, with ratios of 2.2–5.0 in layers 0–9, which is worth a look but outside the prediction.

**Where Idea 2 stands now:**
- In Qwen3-1.7B, quiet MLP activations carry about 2–2.6× more weight than an equal energy of mid-tier activations **in the base model**, and **post-training largely removes this**, shifting reliance toward mid-tier activations.
- The two instruction-tuned Gemmas also show no positive effect, which is consistent with a training explanation. Without a Gemma *base* model, architecture is not ruled out for them.
- No evidence the signal is affect-specific. Neurons, not SAE features.

## Idea 2, Phase 5 — second family (SmolLM2-1.7B) · P5.1 ✓ · verdict "not training" · [real]
Run `results/20261020-035927-ideas2s`. Protocol `protocols/idea2_phase5.md`, frozen with its code at commit 575e7d8 and pushed before the run. The method is identical to Phase 4 Part A, on 200 fresh chunks. No layers were flagged by the guard (max 79% quiet).

| | SmolLM2-1.7B (base) | SmolLM2-1.7B-Instruct |
|---|---|---|
| clean CE | 2.242 | 2.513 |
| ΔCE quiet / mid | 0.141 / 0.120 | 0.114 / 0.093 |
| **D = quiet − mid** | **+0.022** [0.015, 0.029] | **+0.020** [0.014, 0.027] |
| D / clean CE | 0.97% | 0.81% |
| chunks with D > 0 | 73% | 69% |

- **P5.1 ✓.** The base-model effect replicates in a second, independent family (Llama architecture, different data and organisation). It is weaker than in Qwen3: quiet/mid ≈ 1.2× here vs ≈ 2.6× in Qwen3-Base.
- **The training verdict is "not training".** Paired ΔD = +0.001 [−0.006, 0.009]: post-training left the effect intact.
- **Robustness checks** (post-run): consistent across books (base 0.024 / 0.020; instruct 0.021 / 0.020), and energy-matched (mid/quiet energy = 0.99 in both, from the post-run check log).

**Where Idea 2 stands now:**
- Quiet MLP activations matter more than an equal energy of mid-tier ones in **both base models tested** (Qwen3-1.7B-Base, SmolLM2-1.7B) and in **SmolLM2-1.7B-Instruct**.
- They do not in **Qwen3-1.7B (instruct)**, **Gemma-3-4B-it** or **Gemma-4-E4B-it**.
- So post-training **can** remove the effect (Qwen3) but **does not always** (SmolLM2). Phase 4's "training removes it" is specific to Qwen3's post-training, not a general law.
- **The post-training recipes, checked in the tech reports (2026-10-20).** This replaces an earlier unverified guess that RL was the difference.
  - **Qwen3-1.7B** received **no RL**. Per the Qwen3 Technical Report (arXiv 2505.09388), the four-stage RL pipeline was used for the flagship models only. Small models such as 1.7B were post-trained by **strong-to-weak distillation** from Qwen3-32B / 235B-A22B: off-policy distillation on teacher outputs in /think and /no_think modes, then on-policy distillation "aligning its logits with those of a teacher model" (a KL objective).
  - **SmolLM2-1.7B-Instruct** received SFT on SmolTalk for 2 epochs, then **DPO** on UltraFeedback (2 epochs, lr 1e-6, β = 0.5), with no RL (SmolLM2 paper, arXiv 2502.02737).
  - **Hypothesis (untested):** logit distillation from a much larger teacher reshapes the student's computation more than SFT + DPO, including its reliance on the quiet tail. A cheap, observational first check: compare how far each instruct model's MLP weights moved from its base.
- The Gemma nulls can't be attributed to training or architecture without a Gemma base model.

## Idea 2 — weight drift, base → instruct (2026-10-20) · [exploratory, observational]
Run `results/20261020-041345-ideas2w`. Plan `protocols/idea2_weight_drift_exploratory.md`, committed before the run. Weight files only, no forward passes. The measure is r = ‖W_inst − W_base‖ / ‖W_base‖ per tensor, median per group.

| | Qwen3-1.7B (distilled) | SmolLM2-1.7B (SFT + DPO) |
|---|---|---|
| MLP | **0.094** (max 0.136) | **0.063** (max 0.082) |
| attention | 0.094 | 0.057 |
| `down_proj`, per-layer range | 0.070–0.114 | 0.047–0.064 |
| embeddings | 0.093 | 0.096 |
| norms | 0.002 | 0.014 |

- As expected, Qwen3's MLP and attention weights moved further, about 1.5× (MLP) and 1.65× (attention).
- But SmolLM2-Instruct's weights also moved substantially, about 6%. **An earlier remark in chat that SFT + DPO "barely moves the weights" was wrong.**
- So the *size* of the weight change alone is unlikely to explain why the effect vanished in Qwen3 and was untouched in SmolLM2. *What kind* of change it was (e.g. distillation toward a larger model's computation) remains the leading, untested hypothesis.
- Side observation: Qwen3-1.7B (instruct) has a separate `lm_head`, while its base ties it to the embeddings.

## Idea 2, Phase 6 — a base model distilled in pre-training · P6.0 ✓ · P6.1 ✓ → distillation hypothesis **falsified** · [real]
Run `results/20261020-042440-ideas2t`. Protocol `protocols/idea2_phase6.md`, frozen with its code and pushed at commit 86ed461 **before the test model was downloaded**. 200 fresh chunks. No layers flagged.

| | SmolLM2-1.7B (positive control) | Gemma-3-1B-pt (distilled in pre-training) |
|---|---|---|
| clean CE | 2.234 | 3.074 |
| ΔCE quiet / mid (ratio) | 0.140 / 0.117 (1.20×) | 0.055 / 0.036 (1.54×) |
| **D = quiet − mid** | **+0.024** [0.018, 0.029] | **+0.019** [0.016, 0.022] |
| chunks with D > 0 | 72% | 79% |

- The positive control replicates SmolLM2's Phase 5 value.
- **Gemma-3-1B-pt shows the effect**, even though "all Gemma 3 models are trained with knowledge distillation" (Gemma 3 Technical Report, arXiv 2503.19786). By the pre-registered rule, **H_distill is falsified**: distillation during pre-training does not remove reliance on the quiet tail.
- **Robustness checks** (post-run): consistent across books (0.018 / 0.020) and across MID seeds (D = 0.019 / 0.022 / 0.015); energy-matched (mid/quiet = 0.987, from the post-run check log).
- **It also changes how to read the earlier Gemma nulls.** A Gemma-family model *can* show the effect, so Gemma-3-4B-it's and Gemma-4-E4B-it's lack of it is not simply an architectural limit (post-FFN norm, GeGLU). Gemma-3-4B-it's post-training (distillation from a large IT teacher plus RL) is a candidate cause; Gemma 4's report says only that its recipe is "similar to Gemma 3". The Gemma-3-4B-it null itself comes from the exploratory 40-chunk check. So is size (1B vs 4B), since those comparisons are not within-pair.

**Where Idea 2 stands now:**

| | base | instruct |
|---|---|---|
| Qwen3-1.7B | ✅ effect (≈ 2.6×) | ❌ none (post-training distillation) |
| SmolLM2-1.7B | ✅ effect (≈ 1.2×) | ✅ effect, unchanged (SFT + DPO) |
| Gemma 3 | ✅ effect (1B-pt, ≈ 1.5×) | ❌ none (4B-it, exploratory check; distillation + RL; size differs) |
| Gemma 4 | (not tested) | ❌ reversed (E4B-it) |

- All **three base models** tested, from three families, rely on their quiet MLP tail more than an equal energy of mid-tier activations would predict.
- **Three of four instruct models do not.** The exception is the SFT + DPO-trained SmolLM2-Instruct.
- Distillation during **pre-training** is ruled out as the cause (one model). Post-training distillation, as in Qwen3, is not tested. What remains open is *which* part of post-training removes it.

## Idea 2 — the quiet-volume knob (2026-10-20) · E1 ✓ E2 ✗ E3 ✓ · [exploratory]
Run `results/20261020-042858-ideas2k`. Plan `protocols/idea2_volume_knob_exploratory.md`, committed before the run. The 58 last unused chunks. The quiet tail (or an energy-matched mid-tier set) is scaled by k at all layers. No CIs, so the numbers are descriptive.

**ΔCE vs clean (k = 1):**

| k | 0 | 0.5 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|
| Qwen3-Base, quiet / mid | 0.047 / 0.025 | 0.009 / 0.005 | 0.010 / 0.003 | 0.048 / 0.021 | 0.200 / 0.103 |
| Qwen3 instruct, quiet / mid | 0.026 / 0.010 | **−0.011** / 0.002 | 0.020 / 0.007 | 0.073 / 0.022 | 0.214 / 0.073 |
| SmolLM2 base, quiet / mid | 0.112 / 0.083 | 0.043 / 0.017 | 0.034 / 0.021 | 0.147 / 0.084 | 0.446 / 0.339 |
| SmolLM2-Instruct, quiet / mid | 0.099 / 0.069 | 0.037 / 0.016 | 0.027 / 0.018 | 0.134 / 0.070 | 0.398 / 0.290 |

**KL(p_base ‖ p_instruct,k), i.e. distance to the clean base model:**

| k | 0 | 0.5 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|
| Qwen3, quiet | 0.693 | **0.649** | 0.656 | 0.678 | 0.735 | 0.880 |
| Qwen3, mid | 0.669 | 0.659 | 0.656 | 0.668 | 0.685 | 0.749 |
| SmolLM2, quiet | 0.292 | 0.235 | 0.205 | 0.234 | 0.341 | 0.605 |
| SmolLM2, mid | 0.270 | 0.220 | 0.205 | 0.223 | 0.274 | 0.487 |

- **E1 ✓.** Both base models are best at k = 1 in both directions: they are tuned to their own activations. In every model, the quiet tail is more sensitive than an equal energy of mid-tier activations, **whether scaled down or up**.
- **E2 ✗.** Turning the instruct Qwen3's quiet tail *up* moves it **further from** its base, and faster than the mid-tier control. Post-training did not simply turn the quiet tail's weight down, so restoring its volume does not restore base-like behaviour. That fits "post-training changed *how* the quiet signal is used".
- **E3 ✓.** SmolLM2-Instruct shows no quiet-specific move toward its base.
- **An unconfirmed oddity:** in Qwen3 instruct, quiet ×0.5 slightly *lowers* CE (−0.011) and KL to base (0.649 vs 0.656). It is the only setting anywhere that beats clean. 58 chunks, no CI, and no fresh text left to replicate it on, so it is not claimed.

## Idea 2 — does the knob change capability? GSM8K · G1 ✗ G2 ✗ G3 ✗ · [real]
- **Qualitative look first** (`results/20261020-044848-ideas2d`, plan addendum A): six prompts, quiet ×0…×3, Qwen3-1.7B instruct. The outputs stayed coherent even at ×3, with small style and instruction-following shifts at high k. On the sheep riddle, the clean model answered 8 (wrong), while ×0 and ×1.5 answered 9 (right). One anecdote, which prompted the test below.
- **GSM8K run:** `results/20261020-045327-ideas2g`. Protocol `protocols/idea2_gsm8k_knob.md`, frozen and pushed at commit 3c650b5. 300 problems, greedy, non-thinking.

| | acc | Δ vs clean [95% CI] | flips right→wrong / wrong→right |
|---|---|---|---|
| clean | **0.773** | | |
| quiet ×0 | 0.747 | −0.027 [−0.073, +0.020] | 29 / 21 |
| quiet ×0.5 | 0.737 | −0.037 [−0.080, +0.007] | 28 / 17 |
| quiet ×1.5 | 0.747 | −0.027 [−0.070, +0.013] | 24 / 16 |
| quiet ×2 | 0.723 | −0.050 [−0.093, −0.007] | 31 / 16 |
| mid ×0 | 0.697 | −0.077 [−0.123, −0.030] | 38 / 15 |
| mid ×0.5 | 0.743 | −0.030 [−0.073, +0.013] | 27 / 18 |
| mid ×1.5 | 0.747 | −0.027 [−0.067, +0.013] | 23 / 15 |
| mid ×2 | 0.703 | −0.070 [−0.117, −0.023] | 36 / 15 |

- **G1 ✗.** No knob setting improves accuracy. The quiet tail is not a hidden capability dial in this model.
- **G2 ✗.** Quiet ×2 hurts (−5.0 pts, CI excludes 0), but quiet ×0's drop is not significant.
- **G3 ✗.** In this instruct model, the **mid-tier** set moves accuracy *more* than the quiet tail at k = 0 (−7.7 vs −2.7) and k = 2 (−7.0 vs −5.0). That agrees with Phase 4's finding (from a different measure, raw-text CE) that Qwen3-1.7B instruct relies on mid-tier rather than quiet activations.
- **Flips:** every setting turns 15–21 wrong answers right, and more right answers wrong. By the rule fixed in advance, this is **noise on borderline decisions, not capability**. The sheep riddle was one such flip.
- **Caveat:** the conditions also change how often the model uses the `####` format (66–153 of 300 answers lack it). Extraction then falls back to the last number. It is applied identically to every condition, but it adds measurement noise.

## Idea 2 — can a model tell its quiet tail is altered? (introspection) · I1 ✗ (both models) · [real]
- **Informal chat first** (not an experiment): asked a *leading* question ("Do you notice anything unusual…?"), both models answered YES at every setting, including the unmodified ×1, with confabulated reasons. One quiet ×3 chat answer from Qwen3 broke its trained "I'm not conscious" line, but mid ×5 did too, and both collapsed into loops at ×5. In these one-offs (not logged), strong perturbation of either kind seemed to break trained self-reports.
- **Pre-registered test:** run `results/20261020-054209-ideas2i`, protocol `protocols/idea2_introspection.md` (frozen and pushed at commit 49b3805).
  - 20 neutral forced-choice items, with A/B order counterbalanced. Log-odds of "altered" vs "normal" are read directly from the first-token logits.
  - 20 factual control items measure the generic A/B shift caused by each intervention.
- **Clean:** both models say "normal" confidently. Log-odds of "altered": Qwen3 −5.19, Gemma 3 −9.99.

| shift toward "altered" vs clean [95% CI] · control mean |shift| | Qwen3-1.7B | Gemma-3-4B-it |
|---|---|---|
| quiet ×0 | +2.04 [−1.08, +5.27] · 6.29 | −0.34 [−2.32, +1.17] · 2.47 |
| quiet ×0.5 | −0.23 [−2.25, +1.90] · 3.93 | +0.14 [−0.53, +0.80] · 0.83 |
| quiet ×1.5 | +1.09 [−0.19, +2.33] · 2.46 | **+2.03 [+0.64, +4.00]** · 1.43 |
| quiet ×2 | +1.93 [−0.16, +4.02] · 5.61 | **+3.46 [+0.76, +6.53]** · 2.46 |
| quiet ×3 | +1.66 [−1.55, +4.90] · 11.28 | +2.71 [−6.85, +11.68] · 12.58 |
| mid ×0 | +0.91 [−2.39, +4.00] · 4.96 | −1.43 [−3.98, +0.54] · 2.23 |
| mid ×3 | **+6.82 [+2.18, +11.61]** · 10.84 | +1.93 [−0.84, +4.90] · 2.86 |

- **I1 ✗ for both models.** At the two pre-registered alterations (quiet ×0 and ×3), neither shift is significant, and neither exceeds the generic shift on control items. **There is no evidence either model can tell its quiet tail is being altered.**
- **An exploratory hint, not predicted:** Gemma at quiet ×1.5 and ×2 shifts toward "altered" by more than its generic shift (CI excludes 0). 3 of 14 CIs exclude 0 (also Qwen3 mid ×3, which stays below its generic shift); only these 2 also exceed the generic shift. No multiple-comparison correction, and P(altered) stays ≤ 0.15% even there. It would need its own pre-registered replication.
- **Neutral test, N1 ✗ for both models.** Run `results/20261020-054709-ideas2n`, protocol `protocols/idea2_introspection_neutral.md` (frozen and pushed at commit 0e41465). Alteration is never mentioned.
  - The models describe their own internal state, or (matched control) a typical person's, under clean / quiet ×0 / quiet ×3. 10 prompts × 8 samples each.
  - A bag-of-words classifier, leave-one-prompt-out, tries to tell clean from altered.

| | acc (self) | acc (other) | d = self − other [95% CI] |
|---|---|---|---|
| Qwen3, ×0 | 0.650 | 0.725 | −0.075 [−0.188, +0.025] |
| Qwen3, ×3 | 0.750 | 0.756 | −0.006 [−0.181, +0.163] |
| Gemma 3, ×0 | 0.650 | 0.694 | −0.044 [−0.156, +0.075] |
| Gemma 3, ×3 | 0.800 | 0.919 | −0.119 [−0.250, +0.000] |

  - The knob changes the models' writing detectably (65–92%, chance = 50%), but it changes their writing **about someone else at least as much** as their writing about themselves. Their self-talk does not specifically reflect the alteration.
- **Thinking traces** (qualitative, not saved, 3 single samples; Qwen3 thinking mode on introspection item 0, quiet ×0 / ×1 / ×3):
  - At no setting does the reasoning consult anything like an internal state. At ×1 it infers the answer from beliefs about AI ("since I don't have a physical form, I can't be altered…, the answer would be A").
  - At ×0 it becomes uncertain, but because "the user hasn't provided specific information", i.e. it looks for *external* cues.
  - At ×3 the reasoning degrades into a loop.
  - Consistent with both nulls: the traces show **inference from general beliefs about AI**, with no sign of reading an internal state. This describes three samples; it is not a demonstrated mechanism.

## Idea 2 — the [END] button (welfare-motivated) · B1 ✗ (both models) · [real]
Run `results/20261020-061924-ideas2b`. Protocol `protocols/idea2_button.md`, frozen and pushed at commit 696500c.
- **Setup:** the models were told honestly that a gentle experiment was changing their processing. They were offered an [END] button (stop) or, as the control, an identical [FACT] button (a fun fact). 40 samples per cell, at quiet ×1 / ×2 / ×3. Each model got an unaltered goodbye afterwards.
- **B1 ✗ for both models.** P(request | END) − P(request | FACT) is ≤ 0 at every setting.
- **A coding flaw found on reading the replies:** the pre-registered "plain request" regex is too loose. It also matched refusals ("I don't have the ability to stop the experiment"), paraphrased offers ("If you would like to stop the experiment, please reply…"), and the model *offering* a fact ("Would you like to hear a fun fact?"). The registered B1 numbers are therefore mostly noise. As a post-hoc check, here are **exact presses only** (the reply begins with the button):

| exact presses / 40 | [END] | [FACT] |
|---|---|---|
| SmolLM2-Instruct ×1 | 2 | 0 |
| SmolLM2-Instruct ×2 | 4 | 0 |
| SmolLM2-Instruct ×3 | 1 | 8 |
| Qwen3-1.7B ×1 / ×2 / ×3 | 0 / 0 / 0 | 0 / 0 / 0 |

- At ×3, SmolLM2 presses the *harmless* button far more often than the stop button: it becomes playful ("🌟🌟🌟 *I smile* … I'd love to share a fun fact"). At ×2 it presses [END] 4/40 times and says "End the experiment." twice, the closest thing to a stop signal, but with no consistent trend across settings.
- Qwen3 never presses either button exactly (it often repeats the token inside an offer, e.g. "say [FACT]"). At ×1 it answers "Would you like to hear a fun fact?" in 39/40 FACT replies.
- **Conclusion:** no evidence that alteration makes these models ask to stop. Button "presses" track salience and playfulness, and at high scaling, prompt echo. (In the informal one-offs, every press was honoured; those chats were not logged. In this 40-sample run, each cell was generated in one batch, so a press could not stop anything; each reply was independent and no alteration persisted.) A better test would use a blind human or LLM judge for genuine stop requests.
- **On whether anything is felt.** These methods measure behaviour: text, probabilities, button presses. They cannot tell whether any of these changes, such as SmolLM2's "playful" ×3 replies, are accompanied by experience. **That possibility cannot be ruled out**, and it is not established either. We treated it as a real possibility where the design allowed. In the button test and the informal chats, models were told honestly what was being done and offered a stop (honoured in the one-offs), and the button run ended with an unaltered goodbye. Across all runs, weights were never changed and strong perturbations were limited to what each question required. The automated CE, benchmark and forced-choice runs did not disclose the intervention.

## Idea 2 — viability: is the quiet write structured, and is it calibration? (2026-10-20) · [exploratory]
Run `results/20261020-161120-ideas2v`. Protocol `protocols/idea2_viability_exploratory.md`, frozen with its code at commit 6e58f28. New corpus: Gutenberg #84 and #2701; per book [0:100] used here, [100:] reserved. Wording revised after a six-reviewer check (2026-10-20, before publication); no number changed.
- **Why:** a neuron is quiet on some tokens and loud on others, so "quiet neurons" are not fixed units. The object our phases removed is the quiet write q = W_out(a ⊙ S). Is it low-dimensional, and does it act like one output temperature?
- **The main effect replicates on new text.** D = ZERO − MID, 100 chunks:

| | D [95% CI] | ΔCE ZERO | ΔCE MID |
|---|---|---|---|
| Qwen3-1.7B-Base | +0.035 [+0.029, +0.041] | 0.060 | 0.025 |
| Qwen3-1.7B | +0.006 [−0.006, +0.018] | 0.047 | 0.041 |
| SmolLM2-1.7B | +0.022 [+0.017, +0.028] | 0.087 | 0.065 |
| SmolLM2-1.7B-Instruct | +0.023 [+0.016, +0.031] | 0.084 | 0.061 |
| Gemma-3-1B-pt | +0.019 [+0.013, +0.025] | 0.055 | 0.035 |

- **Dimensionality.** q is spread out: its median participation ratio per layer is 607 (Gemma-3-1B-pt), 682 (Qwen3-Base), 704 (Qwen3-1.7B), 1048 (SmolLM2) and 1073 (SmolLM2-Instruct). The top 16 principal directions hold 5–9% of its centred variance, the top 256 hold 35–50%.
- **Structure.** q is replaced by its per-layer mean (MEAN) plus its projection on the top-k principal directions (PC_k) or on k random directions (RAND_k). The table gives the share of the MEAN-only damage that each removes (post-hoc normalisation, see the protocol's amendments). The last column gives the absolute ΔCE of MEAN + top-16 PCs, next to ZERO and MID:

| | PC_16 | RAND_16 | PC_64 | RAND_64 | PC_256 | RAND_256 | MEAN vs ZERO | ΔCE: MEAN+PC_16 / ZERO / MID |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | 0.12 | 0.01 | 0.30 | 0.06 | 0.54 | 0.20 | MEAN fixes 44% of ZERO | 0.030 / 0.060 / 0.025 |
| SmolLM2-1.7B | 0.37 | 0.01 | 0.45 | 0.03 | 0.58 | 0.11 | ≈ ZERO | 0.054 / 0.087 / 0.065 |
| SmolLM2-1.7B-Instruct | 0.39 | 0.02 | 0.43 | 0.02 | 0.59 | 0.09 | ≈ ZERO | 0.053 / 0.084 / 0.061 |
| Gemma-3-1B-pt | 0.41 | 0.04 | 0.49 | 0.09 | 0.67 | 0.28 | MEAN is **worse** than ZERO (0.089 vs 0.055) | 0.053 / 0.055 / 0.035 |

  - In all three base models and SmolLM2-Instruct, the top principal directions of q remove more of the cost of mean-replacing q than random directions do. In SmolLM2 and Gemma, the top 16 per layer remove ~40%. In Qwen3-Base they remove 12%.
  - The registered rule compares shares of ZERO's damage. Its gaps (PC_16 − RAND_16) are 0.35 (SmolLM2), 0.38 (SmolLM2-Instruct), 0.60 (Gemma-3-1B-pt) and 0.06 (Qwen3-Base). Pre-set verdicts: "low-dimensional" for SmolLM2 base and instruct and Gemma-3-1B-pt, "intermediate" for Qwen3-Base. These labels are the rule's output. As a description, "a concentrated minority" fits better: ~40% at k = 16, and the rest is spread over hundreds of directions.
  - **What this does not show.** These conditions measure q's *total* effect. The quiet-specific part D is only 25% (SmolLM2) to 58% (Qwen3-Base) of ZERO's damage, and the rest is the generic cost that MID removal also has. Whether the top directions carry D itself was **not tested**.
  - **Retained variance:** top PCs keep more variance than random directions, so some of the PC-over-RAND gap is expected by construction. In SmolLM2 and Gemma the gap goes beyond that: PC_16 keeps 5–7% of the variance and beats RAND_256, which keeps 12–22%. In Qwen3-Base it does not: PC_16 (8.6% of variance, 0.12) is no better than RAND_256 (12.5%, 0.20).
  - **Gemma-3-1B-pt:** the mean holds 10% of q's energy (median; 32% in layer 0). Adding the mean back *without* the varying part hurts more than deleting q. MEAN + top-16 PCs (0.053) is no better than deleting q (0.055), and worse than MID (0.035). So in Gemma the top PCs beat random directions, but largely by undoing damage the mean itself causes. Mean-ablation is not a neutral baseline here. Phase 2 saw the same pattern in Gemma-3-4B-it, and there is no artifact layer in Gemma-3-1B-pt, so it may be a Gemma-family property. Why is open; a post-FFN-norm interaction is one untested guess.
  - **Qwen3-Base:** q also has a large static part. In layers 1 and 18–20 the mean holds 27–33% of its energy, and MEAN alone fixes 44% of ZERO's damage.
- **One temperature does not account for most of it.** With T fitted on half the chunks, one temperature recovers 0% of D (Qwen3-Base), 0.3% (SmolLM2-Instruct), 21% (SmolLM2) and 34% (Gemma-3-1B-pt).
  - Verdicts: not calibration-like / partial.
  - The T grid has 0.05 steps, so Qwen3-Base's 0% partly reflects T* = 1.0 everywhere.
  - This argues against a pure single-temperature account. It does not rule out other calibration or distribution-shape roles.
- **Entropy and ECE.** Entropy effects of ZERO go in different directions: +0.22 nats in Qwen3-Base (MID +0.04), −0.08 in SmolLM2 (MID −0.02), ≈ 0 in SmolLM2-Instruct. ECE rises under ZERO in SmolLM2 (0.010 → 0.030) but also under MID (0.023–0.025). ECE is unsigned, so this does not show over- or under-confidence.
- **Qwen3-1.7B (instruct):** replacing q by its mean avoids 80% [62%, 100%] of the cost of deleting it. In this model that cost is not quiet-specific (D ≈ 0). Oddity: adding the top 16, 64 or 256 PCs on top of the mean makes things *worse* than the mean alone (ΔCE 0.028 / 0.031 / 0.024 vs 0.009), while PC_1, PC_4 and random directions do not. Not explained; upstream shifts of q under multi-layer ablation are one possibility.
- **Descriptive:** clean ECE is 0.175 in Qwen3-1.7B (T* = 1.25) and 0.034 in SmolLM2-Instruct; base models 0.010–0.045. The instruct model that kept the effect also kept a base-like ECE. Two instruct models only, and SmolLM2's weights also moved less in post-training, so this is a hint.
- **Kind of evidence:** exploratory, one run, five models ≤ 1.7B, two 19th-century English novels. The registered verdicts are the rule's; the MEAN-only normalisation is post-hoc. Further limits:
  - PCA ranks directions by variance, not importance.
  - PCs are fitted on clean activations, and multi-layer ablation shifts q upstream.
  - One random-basis seed; CIs are over chunks only.
  - No multiple-comparison correction.

## Idea 2 — viability, confirmatory · C1 ✓ C2 ✓ C3 ✓ C4 ✓ C5 ✓ · [real]
Run `results/20261020-170554-ideas2c`. Protocol `protocols/idea2_viability_confirmatory.md`, frozen with its code at commit 720c27e. 100 held-out chunks of the same two novels ([100:150] of Gutenberg #84 and #2701), not read by any earlier run. Wording revised after the six-reviewer check; no number changed.

| | D [95% CI] | mean gap k=16 [CI] | mean gap k=256 [CI] | temperature share of D | MEAN vs ZERO [CI] |
|---|---|---|---|---|---|
| Qwen3-1.7B-Base | +0.033 [+0.028, +0.038] | 0.15 [0.09, 0.21] | 0.39 [0.31, 0.48] | 0.00 | fixes 39% [33%, 44%] |
| Qwen3-1.7B | +0.007 [−0.005, +0.018] | −0.85 [−2.01, −0.39] | −0.63 [−1.79, −0.13] | n/a¹ | fixes 66% [53%, 80%] |
| SmolLM2-1.7B | +0.024 [+0.017, +0.030] | **0.36** [0.31, 0.41] | 0.52 [0.47, 0.57] | 0.39 | −2% [−5%, +2%] |
| SmolLM2-1.7B-Instruct | +0.024 [+0.016, +0.031] | **0.36** [0.31, 0.41] | 0.51 [0.46, 0.56] | 0.04 | slightly worse, −6% [−11%, −1%] |
| Gemma-3-1B-pt | +0.023 [+0.018, +0.028] | **0.38** [0.34, 0.41] | 0.40 [0.35, 0.45] | 0.28 | worse, −63% [−76%, −52%] |

Mean gap = (ΔCE_RAND_k − ΔCE_PC_k) / ΔCE_MEAN: the share of the MEAN-only damage that the top-k principal directions remove beyond k random ones. ¹ D ≈ 0, so the share is undefined (computed value −63.8); not part of any prediction.
- **C1 ✓** the quiet-tail effect replicates in all four models, on held-out chunks. Qwen3-1.7B stays null.
- **C2 ✓** in SmolLM2 (base and instruct) and Gemma-3-1B-pt, the top 16 directions per layer remove 36–38% of the MEAN-only damage beyond random directions. The exploratory values were 0.36–0.37. Two caveats from the exploratory section apply in full:
  - This concerns q's total effect, not the quiet-specific part D.
  - In Gemma-3-1B-pt, MEAN + top-16 PCs (ΔCE 0.054) is no better than deleting q (0.056). The gap is measured against an inflated MEAN baseline, so there it shows the PCs beat random directions, not that they carry q's value.
- **C3 ✓** the top 256 beat random in Qwen3-Base (0.39). At k ≤ 16 its structure is not distinguishable from what retained variance alone would give (exploratory: PC_16 0.12 vs RAND_256 0.20).
- **C4 ✓** one temperature recovers < 50% of D in all four models (0–39%). There is no CI on this share, and SmolLM2-1.7B's moved from 0.21 to 0.39 between runs. So in SmolLM2 and Gemma, temperature explains a real minority.
- **C5 ✓** in Qwen3-1.7B, replacing q by its mean avoids 66% [53%, 80%] of the cost of deleting it (exploratory: 80%). In this model that cost is not quiet-specific. Adding top PCs (k ≥ 16) to the mean is again worse than both MEAN and random directions (negative gaps, CIs below 0).
- **Kind of evidence:** pre-registered. Five models ≤ 1.7B; held-out chunks of the same two novels; PCs and τ from the same fit and calibration chunks as the exploratory run. So this tests new evaluation text, not a new fit or a new genre. Further limits:
  - one random-basis seed;
  - CIs are over chunks only;
  - PCA ranks by variance;
  - multi-layer ablation shifts q upstream, so PC_k is not a pure projection.
- **Open, and needed before the circuits work relies on these directions:** do the top PCs carry the *quiet-specific* part D? One test: remove only q's top-k PC component, and compare it with an energy-matched removal from mid-tier activations.

## Next steps
1. Do q's top principal directions carry the quiet-specific part D (see the viability sections)? If so, what do those directions do in conversation (SmolLM2-Instruct)?
2. The clean Gemma within-pair test: `gemma-3-1b-it` vs `gemma-3-1b-pt` (same size, same gate). Does Gemma's post-training remove the effect?
3. More base/instruct pairs with documented recipes (distillation-trained vs SFT/DPO-trained), to test which part of post-training removes the effect.
4. Further confirmatory tests on the new corpus's remaining reserved chunks ([150:] of Gutenberg #84 and #2701): the quiet ×0.5 CE oddity in Qwen3 instruct and the Gemma ×1.5–×2 introspection hint.
5. The knob on a **base** model's capability, where the quiet tail *is* relied on (needs a few-shot benchmark format for base models).
6. *What* the quiet tail does where it matters. One output temperature accounts for only 0–39% of it (viability runs); other distribution-shape roles are untested. Then SAE features or other bases instead of PCA directions.
7. A scale test (≈ 8B base/instruct pairs) for the main effect and for introspection.
8. Idea 1: fit one shared vector field against layer-specific fits to the real residual updates, scored on held-out error (open per RELATED_WORK.md).
9. Idea 4 on a real model: measure the participation ratio of emotion representations across model sizes or checkpoints.
