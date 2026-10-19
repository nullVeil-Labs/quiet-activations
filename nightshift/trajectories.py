"""Idea 1: the last-token residual stream as a trajectory through layers."""
import numpy as np
import torch

from .common import load_model, save_json

# 6 answer groups x 4 paraphrases. Paris/Tokyo share templates on purpose,
# so convergence by meaning can be told apart from convergence by wording.
PROMPTS = {
    " Paris": ["The capital of France is", "France's capital city is",
               "Q: What is the capital of France?\nA:", "The French government is based in the city of"],
    " Tokyo": ["The capital of Japan is", "Japan's capital city is",
               "Q: What is the capital of Japan?\nA:", "The Japanese government is based in the city of"],
    " cold": ["The opposite of hot is", "Ice feels very",
              "Q: What is the opposite of hot?\nA:", "In winter the weather is usually"],
    " blue": ["The color of a clear daytime sky is", "On a sunny day the sky looks",
              "Q: What color is the sky?\nA:", "Roses are red, violets are"],
    " Monday": ["The day after Sunday is", "Sunday is followed by",
                "Q: What day comes after Sunday?\nA:", "The first working day of the week is"],
    " water": ["Fish swim in", "H2O is the chemical formula for",
               "Q: What do fish swim in?\nA:", "Rain is made of"],
}


@torch.no_grad()
def last_token_states(model, tok, text):
    """Residual stream of the final token: h_0 (embeddings) .. h_L (pre-final-norm)."""
    states = []
    hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: states.append(o[0, -1].float()))]
    for layer in model.model.layers:
        hooks.append(layer.register_forward_hook(
            lambda m, i, o: states.append((o[0] if isinstance(o, tuple) else o)[0, -1].float())))
    ids = tok(text, return_tensors="pt").to(model.device)
    model(**ids)
    for h in hooks:
        h.remove()
    return torch.stack(states)  # (L+1, d)


@torch.no_grad()
def logit_lens(model, H):
    logits = model.lm_head(model.model.norm(H.to(model.lm_head.weight.dtype))).float()
    return logits.argmax(-1)


def run(out_dir):
    model, tok = load_model()
    names, groups, H_all, lens_rows = [], [], [], []
    answer_ids = {}
    for g, (answer, prompts) in enumerate(PROMPTS.items()):
        answer_ids[answer] = tok(answer, add_special_tokens=False).input_ids[0]
        for p in prompts:
            H = last_token_states(model, tok, p)
            H_all.append(H.cpu())
            top = logit_lens(model, H).cpu().tolist()
            lens_rows.append({"prompt": p, "answer": answer,
                              "top_tokens": [tok.decode([t]) for t in top],
                              "first_top1_layer": next((l for l, t in enumerate(top) if t == answer_ids[answer]), None)})
            names.append(p)
            groups.append(g)
    H = torch.stack(H_all).numpy().astype(np.float64)  # (P, L+1, d)
    groups = np.array(groups)
    P, L1, _ = H.shape

    # M1 turning: cos between consecutive block writes.
    V = H[:, 1:] - H[:, :-1]
    Vn = V / np.linalg.norm(V, axis=-1, keepdims=True)
    turning = (Vn[:, :-1] * Vn[:, 1:]).sum(-1)  # (P, L-1)
    turning_median = np.median(turning, axis=0)

    # M2 convergence: centered + normalized per layer.
    C = H - H.mean(axis=0, keepdims=True)
    C = C / np.linalg.norm(C, axis=-1, keepdims=True)
    same = groups[:, None] == groups[None, :]
    off = ~np.eye(P, dtype=bool)
    R = []
    for l in range(L1):
        D = np.linalg.norm(C[:, l, None] - C[None, :, l], axis=-1)
        R.append(D[same & off].mean() / D[~same].mean())
    R = np.array(R)

    # Same-template control (Paris vs Tokyo, prompt k vs prompt k): wording held fixed, meaning differs.
    template_pairs = [(k, 4 + k) for k in range(4)]
    template_dist = [float(np.mean([np.linalg.norm(C[a, l] - C[b, l]) for a, b in template_pairs])) for l in range(L1)]
    within_paris_tokyo = [float(np.mean([np.linalg.norm(C[a, l] - C[b, l])
                                         for base in (0, 4) for a in range(base, base + 4)
                                         for b in range(base, base + 4) if a < b])) for l in range(L1)]

    # PCA on layers 1..L for the figure and the lab.
    X = C[:, 1:].reshape(-1, C.shape[-1])
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    proj = ((C[:, 1:] - mu) @ Vt[:2].T)  # (P, L, 2)
    explained = (S[:2] ** 2 / (S ** 2).sum()).tolist()

    mid = slice(1, L1 - 2)
    results = {
        "model": "Qwen3-1.7B-Base", "n_layers": L1 - 1, "n_prompts": P,
        "turning_median_per_layer": turning_median.tolist(),
        "turning_median_layers_1_26": float(np.median(turning[:, mid])),
        "R_per_layer": R.tolist(),
        "R_min_over_l_ge_1": float(R[1:].min()), "R_argmin": int(R[1:].argmin() + 1), "R_1": float(R[1]),
        "same_template_diff_meaning_dist": template_dist,
        "same_meaning_diff_template_dist_paris_tokyo": within_paris_tokyo,
        "logit_lens": lens_rows,
        "pca_explained_variance_2d": explained,
        "pca_paths": proj.tolist(), "groups": groups.tolist(),
        "group_names": list(PROMPTS), "prompts": names,
    }
    results["predictions"] = {
        "P1_curved_median_turning_lt_0.5": results["turning_median_layers_1_26"] < 0.5,
        "P2_attractor_Rmin_lt_0.7_R1": results["R_min_over_l_ge_1"] < 0.7 * results["R_1"],
        "P3_answer_top1_after_layer_14_for_most_groups": None,
    }
    firsts = {}
    for row in lens_rows:
        firsts.setdefault(row["answer"], []).append(row["first_top1_layer"])
    group_first = {a: (min(v for v in ls if v is not None) if any(v is not None for v in ls) else None) for a, ls in firsts.items()}
    results["first_top1_layer_per_group_min"] = group_first
    results["predictions"]["P3_answer_top1_after_layer_14_for_most_groups"] = (
        sum(1 for v in group_first.values() if v is not None and v > 14) > len(group_first) / 2)
    save_json(out_dir / "idea1_trajectories.json", results)
    plot(out_dir, results)
    return results


def plot(out_dir, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = np.array(r["pca_paths"])
    groups = np.array(r["groups"])
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    for i, path in enumerate(paths):
        ax[0].plot(path[:, 0], path[:, 1], color=cmap(groups[i]), alpha=0.7, lw=1)
        ax[0].scatter(path[-1, 0], path[-1, 1], color=cmap(groups[i]), s=20)
    ax[0].set_title("Paths, layers 1-28 (PCA)")
    snap = paths[:, 13]
    ax[1].scatter(snap[:, 0], snap[:, 1], c=[cmap(g) for g in groups])
    ax[1].set_title("Single snapshot, layer 14")
    for g, name in enumerate(r["group_names"]):
        ax[1].scatter([], [], color=cmap(g), label=name.strip())
    ax[1].legend(fontsize=8)
    ax[2].plot(range(1, len(r["turning_median_per_layer"]) + 1), r["turning_median_per_layer"])
    ax[2].axhline(0, color="gray", lw=0.5)
    ax[2].set_title("Turning: median cos(v_l, v_l+1)")
    ax[2].set_xlabel("layer")
    ax[3].plot(r["R_per_layer"], label="R = within / between")
    ax[3].set_title("Convergence by answer group")
    ax[3].set_xlabel("layer")
    ax[3].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "idea1_trajectories.png", dpi=110)
    plt.close(fig)
