# Protocol — Idea 2: viability, confirmatory

Written 2026-10-20, committed together with its code before the run. **Pre-registered.** It tests the exploratory viability result (`protocols/idea2_viability_exploratory.md`, run `results/20261020-161120-ideas2v`) on text no run has read.

## Question
Do these four results replicate on fresh text?
- the quiet-tail effect;
- the quiet write's useful varying part sitting in its top principal directions;
- the failure of one temperature to replace it;
- Qwen3-1.7B's quiet write being mostly a static offset.

## Setup
Identical to the exploratory run (same code, `nightshift/viability.py`, same five models, quiet set, τ, guard, conditions, measures), except:
- **Evaluation chunks:** [100:150] of each shuffled book (Gutenberg #84 and #2701), 100 chunks. No run has read them. [150:] stays reserved.
- τ calibration ([0:20]) and direction fitting ([20:50]) use the same chunks as the exploratory run. They are not test data.
- Per-chunk ΔCE for every condition is saved.

## Measure added for the structure test
- **Mean gap** at k: (ΔCE_RAND_k − ΔCE_PC_k) / ΔCE_MEAN. This is the share of the MEAN-only damage that the top-k principal directions remove beyond k random directions. Bootstrap CI over chunks (ratio of means, 2000 resamples).
- Exploratory values, from the post-hoc view in ANALYSIS.md: k = 16 gives 0.36 (SmolLM2), 0.37 (SmolLM2-Instruct), 0.37 (Gemma-3-1B-pt), 0.11 (Qwen3-Base). k = 256 gives 0.34 for Qwen3-Base.

## Predictions
- **C1 (effect):** D = ZERO − MID has a 95% CI above 0 in Qwen3-1.7B-Base, SmolLM2-1.7B, SmolLM2-1.7B-Instruct and Gemma-3-1B-pt, all four.
- **C2 (low-dimensional):** the mean gap at k = 16 is ≥ 0.20, with CI lower bound > 0, in SmolLM2-1.7B, SmolLM2-1.7B-Instruct and Gemma-3-1B-pt, all three.
- **C3 (Qwen3-Base, more spread out):** the mean gap at k = 256 has a CI lower bound > 0 in Qwen3-1.7B-Base.
- **C4 (not one temperature):** the temperature-recovered share of D (T fitted on even chunks, scored on odd) is < 0.50 in all four models of C1.
- **C5 (static offset):** in Qwen3-1.7B, MEAN recovers ≥ 50% of ZERO's damage (point estimate).

Each prediction is reported as passed or failed, whatever the outcome.

## What would change our reading
- C2 fails → the "few directions" story is not robust, and the conversation-circuits plan should not rely on top PCs.
- C1 fails for a base model → the main effect depends on the corpus.
- C4 fails → a single temperature explains much of the effect after all, in that model.

## Limitations
- Same five small models (≤ 1.7B) and the same genre (19th-century novels) as the exploratory run.
- The same fit chunks define the PCs, so this tests generalisation to new evaluation text, not to a new fit.
- PCA ranks by variance; a different basis might find more structure. That is outside this protocol.

## Note before the run
The added code was smoke-tested on Gemma-3-1B-pt using 4 calibration chunks and 8 fit chunks. No [100:150] chunk was read.
