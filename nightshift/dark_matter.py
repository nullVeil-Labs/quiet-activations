"""Idea 2: are sub-threshold MLP activations structural "dark matter"?"""
import numpy as np
import torch
import torch.nn.functional as F

from .common import load_model, save_json

EMOTIONAL = [
    "When the letter finally arrived, she held it for a long time before opening it, afraid that hope would hurt more than knowing.",
    "He laughed so hard at his daughter's terrible joke that he had to sit down on the kitchen floor, wiping his eyes.",
    "The old dog waited by the door every evening, long after the boy had moved away, and nobody had the heart to tell him.",
    "I am so sorry. I should have listened when you told me you were struggling, and I didn't.",
    "After years of silence, the brothers hugged at the airport, neither of them able to say a single word.",
    "She felt a quiet, aching pride watching her student walk across the stage, remembering how close he came to quitting.",
    "The apartment felt enormous and cold the first night after the funeral.",
    "They danced in the rain outside the hospital because the scan had come back clear.",
    "He kept rereading the message, his chest tight, wondering if he had ruined the only friendship that mattered.",
    "It's okay to be tired. You've been carrying so much, and you don't have to carry it alone tonight.",
    "The crowd went silent, and then someone began to sing, and slowly everyone joined in, voices shaking.",
    "My grandmother's hands smelled of cinnamon and soap, and I would give anything to hold them again.",
]
NEUTRAL = [
    "The bridge is 1.2 kilometres long and carries four lanes of traffic and a separate cycle path.",
    "To reset the router, hold the button on the back panel for ten seconds until the light blinks.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from light.",
    "The meeting has been moved from Tuesday to Thursday at the same time in room 204.",
    "Granite is an igneous rock composed mainly of quartz, feldspar, and small amounts of mica.",
    "The function returns a sorted copy of the list and leaves the original list unchanged.",
    "Trains to the city centre depart every fifteen minutes from platform two.",
    "The recipe calls for two cups of flour, one teaspoon of salt, and half a cup of water.",
    "Mercury is the smallest planet in the solar system and the closest to the Sun.",
    "The library is open from nine in the morning until eight in the evening on weekdays.",
    "Each module contains a short reading, a set of exercises, and a quiz at the end.",
    "The warehouse stores pallets on steel racks that are six metres tall.",
]
TEXTS = EMOTIONAL + NEUTRAL
THRESHOLDS = [0.02, 0.05, 0.1, 0.2]
SEEDS = [0, 1, 2]


def toy(seed=0):
    """Part A: 5 loud features + 50 quiet ones that share a 'tone' direction. True by construction."""
    rng = np.random.default_rng(seed)
    d = 64
    tone = rng.normal(size=d)
    tone /= np.linalg.norm(tone)
    loud = rng.normal(size=(5, d))
    loud /= np.linalg.norm(loud, axis=1, keepdims=True)
    quiet = tone + 2.0 * rng.normal(size=(50, d)) / np.sqrt(d)
    quiet /= np.linalg.norm(quiet, axis=1, keepdims=True)
    a_loud = rng.uniform(0.8, 1.2, 5) * rng.choice([-1, 1], 5)
    a_quiet = rng.uniform(0.10, 0.15, 50)
    x = a_loud @ loud + a_quiet @ quiet
    x_thr = a_loud @ loud  # every quiet coefficient is < 0.2
    # Coherence of the quiet sum vs a random-sign version of it.
    quiet_sum = a_quiet @ quiet
    flipped = (a_quiet * rng.choice([-1, 1], 50)) @ quiet
    denom = np.sqrt((a_quiet ** 2).sum())
    return {
        "tone_readout_full": float(x @ tone), "tone_readout_thresholded": float(x_thr @ tone),
        "cos_full_vs_thresholded": float(x @ x_thr / np.linalg.norm(x) / np.linalg.norm(x_thr)),
        "quiet_share_of_L2sq": float(np.linalg.norm(quiet_sum) ** 2 / np.linalg.norm(x) ** 2),
        "coherence_quiet": float(np.linalg.norm(quiet_sum) / denom),
        "coherence_quiet_sign_shuffled": float(np.linalg.norm(flipped) / denom),
    }


class Ablator:
    """Pre-hook on every down_proj: edits the MLP intermediate activation a."""

    def __init__(self, model):
        self.mode, self.tau, self.gen = None, None, None
        self.stats = []
        self.hooks = [layer.mlp.down_proj.register_forward_pre_hook(self.hook) for layer in model.model.layers]

    def hook(self, module, args):
        if self.mode is None:
            return None
        a = args[0]
        small = a.abs() < self.tau
        if self.mode == "thresh":
            mask = small
        else:  # mass-matched random; "random" includes the crossing entry, "random_under" stops before it
            sq = a.float() ** 2
            target = (sq * small).sum(-1, keepdim=True)
            perm = torch.argsort(torch.rand(sq.shape, generator=self.gen, device="cpu").to(a.device), dim=-1)
            sq_p = torch.gather(sq, -1, perm)
            take_p = (sq_p.cumsum(-1) - sq_p) < target if self.mode == "random" else sq_p.cumsum(-1) <= target
            mask = torch.zeros_like(small).scatter(-1, perm, take_p)
        self.stats.append(((mask.float().mean()).item(),
                           ((a.float() ** 2 * mask).sum() / (a.float() ** 2).sum()).item()))
        return (a.masked_fill(mask, 0),)

    def remove(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def mean_ce(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    logits = model(ids).logits[0, :-1].float()
    return F.cross_entropy(logits, ids[0, 1:]).item()


@torch.no_grad()
def coherence(model, tok, text, tau=0.2, seed=0):
    """Per layer, last position: does the quiet part of W_down @ a add up coherently?"""
    acts = []
    hooks = [l.mlp.down_proj.register_forward_pre_hook(lambda m, a: acts.append(a[0][0, -1].float()))
             for l in model.model.layers]
    model(tok(text, return_tensors="pt").input_ids.to(model.device))
    for h in hooks:
        h.remove()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for layer, a in zip(model.model.layers, acts):
        W = layer.mlp.down_proj.weight.float()  # (d, 6144)
        S = a.abs() < tau
        aS = a * S
        col_sq = (W ** 2).sum(0)
        denom = torch.sqrt((aS ** 2 * col_sq).sum())
        quiet = W @ aS
        loud = W @ (a * ~S)
        signs = torch.randint(0, 2, a.shape, generator=gen).to(a.device) * 2 - 1
        out.append({
            "C": (quiet.norm() / denom).item(),
            "C_sign_shuffled": ((W @ (aS * signs)).norm() / denom).item(),
            "cos_quiet_loud": F.cosine_similarity(quiet, loud, dim=0).item(),
            "frac_entries_quiet": S.float().mean().item(),
        })
    return out


def run(out_dir):
    model, tok = load_model()
    results = {"toy": toy(), "thresholds": THRESHOLDS}
    clean = [mean_ce(model, tok, t) for t in TEXTS]
    abl = Ablator(model)
    rows = []
    try:
        for tau in THRESHOLDS:
            abl.tau = tau
            abl.mode, abl.stats = "thresh", []
            thr = [mean_ce(model, tok, t) - c for t, c in zip(TEXTS, clean)]
            thr_stats = np.mean(abl.stats, axis=0).tolist()
            rand = []
            for seed in SEEDS:
                abl.mode, abl.stats = "random", []
                abl.gen = torch.Generator(device="cpu").manual_seed(seed)
                d = [mean_ce(model, tok, t) - c for t, c in zip(TEXTS, clean)]
                rand.append({"seed": seed, "dCE_mean": float(np.mean(d)),
                             "frac_entries_zeroed": float(np.mean(abl.stats, axis=0)[0]),
                             "frac_L2sq_removed": float(np.mean(abl.stats, axis=0)[1])})
            n = len(EMOTIONAL)
            rows.append({
                "tau": tau,
                "thresh": {"dCE_mean": float(np.mean(thr)), "dCE_emotional": float(np.mean(thr[:n])),
                           "dCE_neutral": float(np.mean(thr[n:])), "per_text": thr,
                           "frac_entries_zeroed": thr_stats[0], "frac_L2sq_removed": thr_stats[1]},
                "random_mass_matched": rand,
            })
    finally:
        abl.remove()
    results["clean_ce_per_text"] = clean
    results["ablation"] = rows
    coh = [coherence(model, tok, t) for t in TEXTS]  # (texts, layers)
    C = np.array([[c["C"] for c in t] for t in coh])
    Cs = np.array([[c["C_sign_shuffled"] for c in t] for t in coh])
    results["coherence"] = {
        "median_C": float(np.median(C)), "median_C_sign_shuffled": float(np.median(Cs)),
        "median_C_per_layer": np.median(C, 0).tolist(), "median_C_shuf_per_layer": np.median(Cs, 0).tolist(),
        "median_cos_quiet_loud_per_layer": np.median([[c["cos_quiet_loud"] for c in t] for t in coh], 0).tolist(),
        "median_frac_entries_quiet_per_layer": np.median([[c["frac_entries_quiet"] for c in t] for t in coh], 0).tolist(),
    }
    r02 = next(r for r in rows if r["tau"] == 0.2)
    results["predictions"] = {
        "P1_thresh_beats_random_every_seed_at_0.2": all(r02["thresh"]["dCE_mean"] > s["dCE_mean"] for s in r02["random_mass_matched"]),
        "P2_median_C_gt_1.2": results["coherence"]["median_C"] > 1.2,
        "P3_emotional_over_neutral_gt_1.2": (r02["thresh"]["dCE_emotional"] / r02["thresh"]["dCE_neutral"]) > 1.2
        if r02["thresh"]["dCE_neutral"] > 0 else None,
    }
    save_json(out_dir / "idea2_dark_matter.json", results)
    plot(out_dir, results)
    return results


def followup(out_dir):
    """Amendment 2 (exploratory): random control that removes <= THRESH's L2 mass."""
    model, tok = load_model()
    clean = [mean_ce(model, tok, t) for t in TEXTS]
    abl = Ablator(model)
    rows = []
    try:
        for tau in THRESHOLDS:
            abl.tau = tau
            abl.mode, abl.stats = "thresh", []
            thr = float(np.mean([mean_ce(model, tok, t) - c for t, c in zip(TEXTS, clean)]))
            thr_l2 = float(np.mean(abl.stats, axis=0)[1])
            under = []
            for seed in SEEDS:
                abl.mode, abl.stats = "random_under", []
                abl.gen = torch.Generator(device="cpu").manual_seed(seed)
                d = float(np.mean([mean_ce(model, tok, t) - c for t, c in zip(TEXTS, clean)]))
                st = np.mean(abl.stats, axis=0)
                under.append({"seed": seed, "dCE_mean": d, "frac_entries_zeroed": float(st[0]), "frac_L2sq_removed": float(st[1])})
            rows.append({"tau": tau, "thresh_dCE_mean": thr, "thresh_frac_L2sq_removed": thr_l2, "random_under": under})
    finally:
        abl.remove()
    results = {"note": "Exploratory (protocol amendment 2). Not a confirmatory test.", "rows": rows,
               "random_under_still_costlier_at_0.2_every_seed": all(
                   s["dCE_mean"] > rows[-1]["thresh_dCE_mean"] for s in rows[-1]["random_under"])}
    save_json(out_dir / "idea2_followup_under_matched.json", results)
    return results


def plot(out_dir, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    taus = [row["tau"] for row in r["ablation"]]
    ax[0].plot(taus, [row["thresh"]["dCE_mean"] for row in r["ablation"]], "o-", label="threshold |a|<tau")
    ax[0].plot(taus, [np.mean([s["dCE_mean"] for s in row["random_mass_matched"]]) for row in r["ablation"]],
               "s--", label="random, same L2 mass")
    ax[0].set_xlabel("tau")
    ax[0].set_ylabel("ΔCE (nats)")
    ax[0].set_title("Deleting quiet activations")
    ax[0].legend()
    ax[1].plot([row["thresh"]["frac_entries_zeroed"] for row in r["ablation"]], "o-", label="entries zeroed")
    ax[1].plot([row["thresh"]["frac_L2sq_removed"] for row in r["ablation"]], "s-", label="L2² removed")
    ax[1].set_xticks(range(len(taus)), [str(t) for t in taus])
    ax[1].set_title("How much is 'quiet'")
    ax[1].legend()
    ax[2].plot(r["coherence"]["median_C_per_layer"], label="C (real)")
    ax[2].plot(r["coherence"]["median_C_shuf_per_layer"], label="C (signs shuffled)")
    ax[2].axhline(1, color="gray", lw=0.5)
    ax[2].set_xlabel("layer")
    ax[2].set_title("Coherence of the quiet sum")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "idea2_dark_matter.png", dpi=110)
    plt.close(fig)
