# Protocol — Idea 4: empathy as compression (MDL)

Frozen: 2026-10-19, before any run.

## Question
When predicting how people respond to situations, is a small latent "state vector" model a shorter description (two-part MDL code) than a table of rules? Under what conditions?

## Worlds (synthetic)
- Situations: 16 binary cues (2^16 = 65,536 possible situations). Responses: 8 categories.
- **LATENT world:** cues → 3-d hidden state s = tanh(A·cues) → response ~ softmax(B·s · 3).
- **RULE world (control):** each situation's response distribution is drawn independently (a random lookup table, Dirichlet α = 0.3). No low-dimensional structure.
- Training sizes N ∈ {200, 1000, 5000, 20000}. Test set: 5000 fresh samples. Seeds 0, 1, 2.

## Models
- **RULES:** a table of seen situations with add-½ smoothed counts. Unseen situations fall back to the global marginal. Model cost: 16 bits per stored situation + (8−1)·½·log2(N) bits for its distribution.
- **LATENT:** logistic model cues → k-d tanh → softmax, k = 3, fit by gradient descent on training NLL. Model cost: (#params)·½·log2(N) bits (standard two-part / BIC precision).
- **MARGINAL:** a single distribution (7 params).

## Measures
- Total code length L = L(model) + L(data | model) in bits, on training data.
- Held-out NLL in bits per sample.
- **Steering:** how many parameters must change to shift every situation's response distribution towards response 0 by about the same amount. LATENT: k (move s via a bias). RULES: every stored row. Measured, not assumed: report the edit count and the achieved mean shift.

## Predictions
- P1: in the LATENT world, LATENT has the smallest L at every N ≥ 1000, and the best held-out NLL.
- P2 (control): in the RULE world, LATENT does *not* beat RULES on held-out NLL at N = 20000.
- P3: steering LATENT takes ≤ 3 parameter changes, and RULES takes a number of edits equal to its stored rows.

## Caveats (important)
- Whether the latent model wins depends on whether the world has low-dimensional structure. The toy builds that structure in, and the control removes it. The toy can't say whether human psychology is low-dimensional. That is an empirical question.
- Compression explains why a predictor might *represent* others' states compactly. It says nothing about whether the model *has* feelings. "Functional empathy" here means only "a compact latent model of others' states".
