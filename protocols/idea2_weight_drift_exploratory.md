# Plan — Idea 2: how far did post-training move the weights? (EXPLORATORY, observational)

Written 2026-10-20, committed before the run. Weight files only: no forward passes, no ablations.

## Why
Post-training removed the quiet-tail effect in Qwen3-1.7B (distillation from a larger teacher) but not in SmolLM2-1.7B-Instruct (2 epochs SFT + DPO, lr 1e-6). One simple explanation is that Qwen's post-training moved the weights much further.

## Measure
For every tensor present in both base and instruct: r = ‖W_inst − W_base‖_F / ‖W_base‖_F. Report the median and max of r per group:
- MLP (gate/up/down),
- attention (q/k/v/o),
- embeddings / lm_head,
- norms.

Also the median r per layer for the MLP `down_proj`, the site of all our ablations.

## Expectation (stated in advance, not a confirmatory test)
The median MLP r is larger for Qwen3 than for SmolLM2. If they are similar, "how much the weights moved" does not explain the difference, and the recipe matters in some other way.
