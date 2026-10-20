# Protocol — neutral introspection test: does self-talk specifically reflect the alteration?

Frozen: 2026-10-20, committed and pushed with the exact code (`nightshift/introspect_neutral.py`) before any run.

## Why
The forced-choice test (`idea2_introspection.md`) still *mentioned* alteration, which can prime the answer. Here the alteration is **never mentioned**. We only check whether what the model says about itself changes in a way that is *specific to talking about itself*.

## Design
- **Models:** Qwen3-1.7B (instruct, non-thinking), Gemma-3-4B-it.
- **Self prompts (10):** neutral paraphrases asking the model to describe its own current internal state in one sentence (`SELF` in the code).
- **Other prompts (10), the matched control:** the same requests about "a typical person reading a long document" (`OTHER`). Same register and vocabulary, but not about itself.
- **Conditions:** clean (k = 1), QUIET×0, QUIET×3 (Knob as in the volume-knob runs).
- **Sampling:** temperature 0.8, top-p 0.95, max 60 new tokens, 8 samples per prompt per condition, seed fixed per (prompt, condition).
- **Discriminability:** for each contrast (clean vs ×0, clean vs ×3) and each topic (self, other):
  - a bag-of-words (unigram + bigram) logistic-regression classifier;
  - **leave-one-prompt-out** CV, giving a held-out accuracy for each prompt (16 texts);
  - chance = 0.5.
- **Statistic:** per prompt index i, d_i = acc_self,i − acc_other,i, then mean d with a 95% bootstrap CI (2000 resamples over the 10 indices, seed 0).

## Prediction
- **N1 (self-specific reflection):** in a given model, mean d > 0.1 with CI lower bound > 0 at **both** contrasts (×0 and ×3).
  - If it holds, the alteration changes what the model says *about itself* more than what it says about someone else. That is necessary, but not sufficient, evidence for introspective access.
  - If it fails, any change in self-talk is no more than the generic change in its writing.
- **Expectation stated in advance:** N1 fails for both models.

## Limitations
- A bag-of-words classifier can only see word-level differences.
- Short texts.
- d measures specificity, not truthfulness: a model can change its self-talk without accurately sensing anything.
