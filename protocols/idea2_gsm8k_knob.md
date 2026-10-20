# Protocol — Idea 2: does the quiet-volume knob change capability? (GSM8K)

Frozen: 2026-10-20, committed with the exact code (`nightshift/knob.py`, `gsm8k`) before any evaluation question was run.

## Why
The qualitative look (knob addendum A, run 20261020-044848) suggested that scaling the quiet tail can **flip borderline decisions**. On one riddle, the normal model was wrong while quiet ×0 and ×1.5 were right. That's one anecdote. This protocol tests whether the knob changes **capability** on a standard benchmark with a human-written answer key, and whether any change is specific to the quiet tail.

## Data
- GSM8K test set (OpenAI, MIT licence), `data/gsm8k_test.jsonl` from `github.com/openai/grade-school-math`, sha256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`, 1319 problems.
- **300 problems**, drawn with `random.Random(0).sample(range(1319), 300)`. The smoke test uses only problems outside that subset.
- Gold answer: the number after `####` in the answer field.

## Model and decoding
- Qwen3-1.7B (instruct), revision `70d244cc`. Chat template with `enable_thinking=False`.
- User message: the question + "\nGive the final answer as a number after '####'."
- Greedy decoding, max 400 new tokens, batched with left padding.
- **Answer extraction:** the number after the last `####` if present, otherwise the last number in the output. Commas are removed, then the value is compared numerically with the gold answer (tolerance 1e-6).

## Conditions
The same Knob as the volume-knob run: outlier-excluded quiet set, per-layer τ_l on the same 40 calibration chunks, > 95% guard, all layers.
- **QUIET×k**, k ∈ {0, 0.5, 1.5, 2}
- **MID×k**, k ∈ {0, 0.5, 1.5, 2} (energy-matched mid-tier set, seed 0)
- **k = 1** (clean), shared by both.

## Measures
- Accuracy per condition.
- Paired difference vs clean, with a 95% bootstrap CI (2000 resamples over problems, seed 0).
- Flip counts vs clean: wrong→right and right→wrong.

## Predictions
- **G1 (capability gain, the hypothesis of interest):** some QUIET×k beats clean, with CI lower bound > 0. If it holds, also check whether the gain is quiet-specific, i.e. whether MID×k at the same k gains less.
- **G2 (expected default):** QUIET×0 and QUIET×2 are both *below* clean (CI upper bound < 0). The model is tuned to k = 1.
- **G3 (quiet sensitivity):** at each k, |acc(QUIET×k) − clean| ≥ |acc(MID×k) − clean|, i.e. the quiet tail moves capability at least as much as mid-tier, in ≥ 3 of the 4 k values.
- **Fixed in advance:** many flips in *both* directions with no net gain means the knob adds noise to borderline decisions rather than capability.

## Limitations
- One model, one benchmark, greedy decoding only, 300 problems.
- The answer-extraction heuristic could miscount unusual formats. It is applied identically to every condition.
