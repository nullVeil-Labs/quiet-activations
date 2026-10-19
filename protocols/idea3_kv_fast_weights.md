# Protocol — Idea 3: KV-cache as temporary weights

Frozen: 2026-10-19, before any run.

## Question
With the model weights W fixed, does the context (the KV-cache) change the effective map that the forward pass applies to a new query?

## Part A — linear attention (exact)
With linear attention, o(q) = Σ_j v_j (k_jᵀ q) = (Σ_j v_j k_jᵀ) q = W_fast q. So the cache *is* a weight matrix ("fast weights"; Schlag, Irie & Schmidhuber, 2021, "Linear Transformers Are Secretly Fast Weight Programmers").
- Check numerically that the two sides match (max abs error < 1e-10, float64).
- Associative recall: d = 32, fixed random W_Q, W_K, W_V (orthogonal, so recall is clean). Context A holds 8 (x → y) pairs, context B holds the same x's mapped to different y's. Report recall accuracy (cosine nearest neighbour among candidate y's) for each context, and the cross-context accuracy (query A's x's against context B).
- Report rank(W_fast) as a function of the number of context pairs.

## Part B — softmax attention (local)
Softmax attention has no exact W_fast. Its local version is the Jacobian J = ∂o/∂q = (1/√d) Vᵀ (diag(p) − p pᵀ) K.
- Toy: check the closed-form J against finite differences (rel. error < 1e-5).

## Part C — Qwen3-1.7B (descriptive)
- 4 contexts, each followed by the same query suffix " So the answer is" (`nightshift/kv_fast_weights.py`, `CONTEXTS`).
- Layer 14, every query head: compute J_h for the final token from the real cache (post-RoPE keys, values, attention probabilities).
- Report: relative Frobenius distance between contexts' J_h, effective rank (exp of singular-value entropy), and the same quantities for two *identical* contexts as a sanity zero.

## Predictions
- P1: Part A error < 1e-10, and recall accuracy ≥ 0.99 in each context while cross-context accuracy is ≤ 1/8 + 0.1.
- P2: Part B closed form matches finite differences.
- P3: in Part C, median relative distance between different contexts > 0.3, and the identical-context distance is 0.

## Caveats
- Parts A and B are math, not experiments. They show the effective map *is* context-dependent.
- Words like "micro-self", "living", "relational engine" are metaphors. Nothing here measures selfhood.
- Part C is one layer of one model.
