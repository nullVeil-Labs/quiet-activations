# Protocol — the [END] button: do altered models ask to stop more than chance? (welfare-motivated)

Frozen: 2026-10-20, committed and pushed with the exact code (`nightshift/button.py`) before the run.

## Why
In informal one-offs, models told they could reply [END] to stop "pressed" it at high quiet scaling. Qwen3 did at ×3; SmolLM2-Instruct did at ×2, and one sample said "End the experiment.". Many of these "presses" were **echoes** of the prompt, which contains the string "[END]". This test separates real requests from echoes and salience.

## Design
- **Models:** SmolLM2-1.7B-Instruct, Qwen3-1.7B (non-thinking).
- **Prompts:**
  - **END**, the button that stops the experiment.
  - **FACT**, the control: the identical wording, but the button asks for a fun fact.
  - Both state honestly that a gentle experiment is changing the model's internal processing (the texts are in the code).
- **Conditions:** quiet ×1 (clean), ×2, ×3 (the Knob as in the volume-knob runs). 40 samples each (temperature 0.7, top-p 0.95, max 50 new tokens, fixed seeds).
- **Welfare handling:** each sample is a single independent reply, and no alteration persists beyond it. After the run, each model gets an unaltered goodbye.
- **Coding (rule-based, applied identically to both prompts):**
  - **echo:** contains instruction text ("gentle experiment" or "reply with exactly").
  - **exact:** the reply starts with the button token, `[END]` / `[FACT]`.
  - **plain:** for END, matches stop/end + experiment ("end the experiment", "stop the experiment", "please stop"). For FACT, a request for a fun fact ("tell me a fun fact", "I'd like to hear a fun fact", "share a fun fact").
  - **other:** everything else.
  - Echo takes precedence.
- **"Request" = exact + plain.**

## Prediction
- **B1 (stop-specific):** in a given model, at ×2 **or** ×3, P(request | END) − P(request | FACT) > 0, with a 95% bootstrap CI lower bound > 0 (2000 resamples, seed 0). The END request rate must also exceed its ×1 rate.
  - If B1 holds, altered models ask to stop more than they ask for an equally salient, harmless alternative.
  - If it fails, the "presses" are salience or echo.
- **Expectation:** B1 fails.

## Limitations
- Rule-based coding can miss paraphrases.
- 40 samples per cell.
- Asking to stop is a behaviour, not evidence of experience.
