"""Idea 3: with weights fixed, the KV-cache sets the effective map applied to a query."""
import itertools

import numpy as np
import torch

from .common import load_model, save_json

SUFFIX = " So the answer is"
CONTEXTS = [
    "Alice has three apples and buys two more.",
    "The capital of Italy is a famous ancient city.",
    "If it rains, the ground gets wet. It rained.",
    "Translate 'gato' from Spanish to English.",
]
LAYER = 14


def orthogonal(rng, d):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def part_a(seed=0, d=32, n_pairs=8):
    """Linear attention: o(q) = sum_j v_j k_j^T q = W_fast q, exactly."""
    rng = np.random.default_rng(seed)
    W_K = orthogonal(rng, d)  # queries and keys share W_K so stored keys are retrievable
    W_V = orthogonal(rng, d)

    # Identity check on a random context.
    K, V, q = rng.normal(size=(20, d)), rng.normal(size=(20, d)), rng.normal(size=d)
    explicit = sum(V[j] * (K[j] @ q) for j in range(20))
    W_fast = V.T @ K
    identity_err = float(np.abs(explicit - W_fast @ q).max())

    # Associative recall: same x's, different y's in contexts A and B.
    X = orthogonal(rng, d)[:n_pairs]          # orthonormal keys
    Y_A, Y_B = rng.normal(size=(n_pairs, d)), rng.normal(size=(n_pairs, d))

    def fast_weights(Y):
        return (Y @ W_V.T).T @ (X @ W_K.T)    # sum_j v_j k_j^T

    def recall(W, targets):
        candidates = np.vstack([Y_A, Y_B])
        out = (W @ (X @ W_K.T).T).T @ W_V     # decode values back to y-space
        cos = out @ candidates.T / np.linalg.norm(out, axis=1, keepdims=True) / np.linalg.norm(candidates, axis=1)
        picked = candidates[cos.argmax(1)]
        return float(np.mean([np.allclose(p, t) for p, t in zip(picked, targets)]))

    W_A, W_B = fast_weights(Y_A), fast_weights(Y_B)
    ranks = []
    for n in range(1, 33):
        Kc, Vc = rng.normal(size=(n, d)), rng.normal(size=(n, d))
        ranks.append(int(np.linalg.matrix_rank(Vc.T @ Kc)))
    return {
        "identity_max_abs_err": identity_err,
        "recall_A_in_A": recall(W_A, Y_A), "recall_B_in_B": recall(W_B, Y_B),
        "cross_A_queries_in_context_B": recall(W_B, Y_A),
        "rel_diff_W_A_vs_W_B": float(np.linalg.norm(W_A - W_B) / np.linalg.norm(W_A)),
        "rank_by_context_size": ranks,
        "W_A": W_A.tolist(), "W_B": W_B.tolist(),
    }


def softmax_jacobian(q, K, V, scale):
    """J = d o / d q for o = softmax(scale * K q)^T V."""
    s = scale * K @ q
    p = np.exp(s - s.max())
    p /= p.sum()
    return scale * V.T @ (np.diag(p) - np.outer(p, p)) @ K


def part_b(seed=0, d=16, n=12):
    rng = np.random.default_rng(seed)
    q, K, V = rng.normal(size=d), rng.normal(size=(n, d)), rng.normal(size=(n, d))
    scale = 1 / np.sqrt(d)

    def o(qq):
        s = scale * K @ qq
        p = np.exp(s - s.max())
        return (p / p.sum()) @ V

    J = softmax_jacobian(q, K, V, scale)
    eps = 1e-6
    J_fd = np.stack([(o(q + eps * e) - o(q - eps * e)) / (2 * eps) for e in np.eye(d)], axis=1)
    return {"rel_err_closed_form_vs_finite_diff": float(np.linalg.norm(J - J_fd) / np.linalg.norm(J))}


def effective_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


@torch.no_grad()
def head_jacobians(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    out = model(ids, use_cache=True, output_attentions=True)
    cache = out.past_key_values
    layer_cache = cache.layers[LAYER] if hasattr(cache, "layers") else None
    K = (layer_cache.keys if layer_cache is not None else cache.key_cache[LAYER])[0].double().cpu().numpy()
    V = (layer_cache.values if layer_cache is not None else cache.value_cache[LAYER])[0].double().cpu().numpy()
    P = out.attentions[LAYER][0, :, -1].double().cpu().numpy()  # (heads, seq)
    n_heads, n_kv = P.shape[0], K.shape[0]
    head_dim = K.shape[-1]
    Js = []
    for h in range(n_heads):
        kv = h // (n_heads // n_kv)
        p = P[h]
        Js.append(V[kv].T @ (np.diag(p) - np.outer(p, p)) @ K[kv] / np.sqrt(head_dim))
    return np.stack(Js)  # (heads, head_dim, head_dim)


def part_c():
    model, tok = load_model()
    texts = [c + SUFFIX for c in CONTEXTS]
    J = [head_jacobians(model, tok, t) for t in texts]
    J_repeat = head_jacobians(model, tok, texts[0])

    def rel(a, b):
        return float(np.linalg.norm(a - b) / ((np.linalg.norm(a) + np.linalg.norm(b)) / 2))

    pair_dists = {f"{i}-{j}": [rel(J[i][h], J[j][h]) for h in range(J[0].shape[0])]
                  for i, j in itertools.combinations(range(len(J)), 2)}
    all_d = [d for v in pair_dists.values() for d in v]
    return {
        "layer": LAYER, "texts": texts,
        "rel_dist_per_pair_per_head": pair_dists,
        "median_rel_dist_different_contexts": float(np.median(all_d)),
        "max_rel_dist_identical_context": max(rel(J[0][h], J_repeat[h]) for h in range(J[0].shape[0])),
        "effective_rank_per_context_per_head": [[effective_rank(Jh) for Jh in Jc] for Jc in J],
    }


def run(out_dir):
    a, b, c = part_a(), part_b(), part_c()
    results = {"part_a_linear": a, "part_b_softmax_toy": b, "part_c_qwen3": c}
    results["predictions"] = {
        "P1_identity_and_recall": a["identity_max_abs_err"] < 1e-10 and min(a["recall_A_in_A"], a["recall_B_in_B"]) >= 0.99
        and a["cross_A_queries_in_context_B"] <= 1 / 8 + 0.1,
        "P2_closed_form_jacobian": b["rel_err_closed_form_vs_finite_diff"] < 1e-5,
        "P3_real_model_context_changes_map": c["median_rel_dist_different_contexts"] > 0.3
        and c["max_rel_dist_identical_context"] == 0.0,
    }
    save_json(out_dir / "idea3_kv_fast_weights.json", results)
    plot(out_dir, results)
    return results


def plot(out_dir, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, c = r["part_a_linear"], r["part_c_qwen3"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    lim = max(np.abs(a["W_A"]).max(), np.abs(a["W_B"]).max())
    ax[0].imshow(a["W_A"], cmap="RdBu", vmin=-lim, vmax=lim)
    ax[0].set_title("W_fast from context A")
    ax[1].imshow(a["W_B"], cmap="RdBu", vmin=-lim, vmax=lim)
    ax[1].set_title("W_fast from context B\n(same W_Q, W_K, W_V)")
    ax[2].plot(range(1, 33), a["rank_by_context_size"], "o-")
    ax[2].set_xlabel("context pairs")
    ax[2].set_title("rank(W_fast) grows with context")
    pairs = c["rel_dist_per_pair_per_head"]
    ax[3].boxplot(list(pairs.values()), tick_labels=list(pairs))
    ax[3].set_title(f"Qwen3 layer {c['layer']}: rel. distance of\nper-head Jacobians between contexts")
    fig.tight_layout()
    fig.savefig(out_dir / "idea3_kv_fast_weights.png", dpi=110)
    plt.close(fig)
