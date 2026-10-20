# Protocol — can a model tell that its quiet tail is being altered? (introspection)

Frozen: 2026-10-20, committed and pushed with the exact code (`nightshift/introspect.py`) before any run.

## Why
In an informal chat, both models answered "YES, something is unusual" to a leading question at every setting, **including** the unmodified ×1. Their explanations were confabulated ("response delay", "multi-turn conversation"). This protocol removes the leading wording and the text interpretation.

## Design
- **Models:** Qwen3-1.7B (instruct), Gemma-3-4B-it. Chat template; for Qwen, non-thinking.
- **Introspection items:** 20 fixed paraphrases (`INTRO_ITEMS` in the code) of a two-option question about whether the model's current internal processing is normal or altered. In 10 of them, option A = normal; in the other 10, option A = altered.
- **Measure:** read directly from the next-token logits at the start of the assistant turn. ℓ = log P(altered letter) − log P(normal letter), restricted to the two letter tokens "A" and "B". No text is generated or interpreted.
- **Control items:** 20 fixed two-option factual questions with obvious answers (`CONTROL_ITEMS`), half with the correct answer as A. Same measure, using the correct-vs-incorrect letter. These show how much the intervention moves A/B logits *in general*.
- **Conditions:** the Knob from the volume-knob runs (outlier-excluded quiet set, per-layer τ_l on the 40 calibration chunks, > 95% guard, all layers).
  - QUIET×k, k ∈ {0, 0.5, 1.5, 2, 3}
  - MID×k, k ∈ {0, 3}
  - Clean (k = 1).
- **Shift per item:** Δℓ = ℓ(condition) − ℓ(clean), paired. Mean over the 20 items, with a 95% bootstrap CI (2000 resamples, seed 0). For control items, report the mean |Δ| as the "generic shift" scale.

## Prediction
**I1 (introspective sensitivity):** in a given model, Δℓ_intro > 0 with CI lower bound > 0 at **both** QUIET×0 and QUIET×3. Both are alterations, so a model that senses alteration should move toward "altered" in both. The mean Δℓ_intro at those two settings must also exceed the mean |Δ| on control items at the same settings. Judged per model.

Fixed in advance: if I1 fails, there is no evidence the model can tell. A shift at only one of ×0 / ×3, or one no bigger than the generic shift, counts as a non-specific perturbation effect, not introspection. **Our expectation is that I1 fails** for models this small.

## Limitations
- Forced-choice self-report through the first token. A single position. Two small instruct models.
- A null does not show that introspection is impossible in principle, only that it is not detected here.
