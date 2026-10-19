# Protocol — Idea 2: sub-threshold "dark matter"

Frozen: 2026-10-19, before any run.

## Question
Do the small MLP activations that a threshold would delete matter more than their energy suggests, and do their contributions add up coherently ("interfere constructively")?

## Part A — toy (shows what's possible)
A vector built from 5 large features (|a| ≈ 1) plus 50 small ones (|a| ∈ [0.10, 0.15]), all sharing a common "tone" component, in d = 64 with random unit feature directions. Threshold at 0.2. Report the cosine of the tone readout before and after thresholding.
This toy is true by construction. It shows the mechanism can exist, not that LLMs use it.

## Part B — Qwen3-1.7B (what actually happens)
- Site: the MLP intermediate activation a = silu(gate)·up (6144-d) at every layer, all token positions.
- Text: 24 short passages written for this repo (`nightshift/dark_matter.py`, `TEXTS`), half emotional and half neutral.
- Thresholds τ ∈ {0.02, 0.05, 0.1, 0.2}.
- Ablations, applied to all layers at once:
  - **THRESH**: zero every entry with |a| < τ.
  - **MASS-MATCHED RANDOM**: at each layer and position, zero a random set of entries whose removed squared L2 norm matches THRESH. Entries are drawn in random order, and the draw stops at the first entry that crosses the target.
- Metrics: mean next-token cross-entropy change ΔCE vs clean, fraction of entries zeroed, fraction of L2² removed.
- **Coherence** (per layer, last position, clean run): c_i = a_i·W_down[:, i] for the small set S = {i : |a_i| < 0.2}. C = ‖Σ_S c_i‖ / sqrt(Σ_S ‖c_i‖²). Random phases give C ≈ 1, constructive interference gives C > 1, destructive gives C < 1. Also compute cos(Σ_S c_i, Σ_large c_i).
- Seeds: random ablation uses seeds 0, 1, 2. Report the mean and the range.

## Predictions
- P1 (dark matter matters per unit energy): at τ = 0.2, ΔCE(THRESH) > ΔCE(RANDOM) with every seed.
- P2 (constructive interference): median C over layers > 1.2.
- P3 (tone): ΔCE(THRESH) on emotional texts / ΔCE on neutral texts > 1.2.
- Null for each: the inequality doesn't hold.

## Caveats
- Neurons are not SAE features. The hypothesis is about features, so this is an approximation using neurons.
- 24 passages, one model, no significance testing.

## Amendment 1 — 2026-10-19, added before the first run
- Exploratory control, not a prediction: a sign-shuffled baseline C_shuf. It is C recomputed after randomly flipping the signs of a_S, which keeps the magnitudes but breaks the phase alignment. P2 is judged against the fixed 1.2 as written. C_shuf is reported next to it so readers can see how far C would sit above 1 by chance.

## Amendment 2 — 2026-10-19, written AFTER run 20261019-191428 (exploratory)
In that run the "mass-matched" random control removed about 1.5× more L2² than THRESH: 6.7–6.9% vs 4.5% at τ = 0.2. The cause is the frozen rule that includes the crossing entry, which is sometimes a large outlier. The pre-registered P1 verdict for that run stands as recorded.
Exploratory follow-up (`python run.py 2f`): the same comparison, but the random draw stops *before* the crossing entry, so it removes ≤ THRESH's L2². This tilts the comparison in THRESH's favour. If random ablation still costs more than THRESH, the run-1 conclusion is robust to the mismatch. This follow-up is not a confirmatory test.
