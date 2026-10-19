"""Idea 4: two-part MDL of a rule table vs a small latent-state model."""
import numpy as np
import torch

from .common import save_json

N_CUES, N_RESP, K = 16, 8, 3
SIZES = [200, 1000, 5000, 20000]
SEEDS = [0, 1, 2]
N_TEST = 5000


class World:
    def __init__(self, kind, seed):
        rng = np.random.default_rng(1000 + seed)
        self.kind, self.rng = kind, np.random.default_rng(seed)
        if kind == "latent":
            self.A = rng.normal(size=(K, N_CUES)) * 0.6
            self.a0 = rng.normal(size=K) * 0.3
            self.B = rng.normal(size=(N_RESP, K))
        else:
            self.table = rng.dirichlet([0.3] * N_RESP, size=2 ** N_CUES)

    def probs(self, cues):
        if self.kind == "latent":
            s = np.tanh(cues @ self.A.T + self.a0)
            z = 3 * s @ self.B.T
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            return p / p.sum(1, keepdims=True)
        return self.table[index(cues)]

    def sample(self, n):
        cues = self.rng.integers(0, 2, size=(n, N_CUES)).astype(float)
        p = self.probs(cues)
        y = np.array([self.rng.choice(N_RESP, p=row) for row in p])
        return cues, y


def index(cues):
    return (cues.astype(np.int64) * (1 << np.arange(N_CUES))).sum(1)


def bits(p, y):
    return -np.log2(np.clip(p[np.arange(len(y)), y], 1e-12, None))


class Rules:
    def fit(self, cues, y):
        self.n = len(y)
        self.marg = (np.bincount(y, minlength=N_RESP) + 0.5) / (len(y) + 0.5 * N_RESP)
        self.counts = {}
        for i, r in zip(index(cues), y):
            self.counts.setdefault(i, np.zeros(N_RESP))[r] += 1
        self.rows = {i: (c + 0.5) / (c.sum() + 0.5 * N_RESP) for i, c in self.counts.items()}
        return self

    def predict(self, cues):
        return np.array([self.rows.get(i, self.marg) for i in index(cues)])

    def model_bits(self):
        per_dist = (N_RESP - 1) * 0.5 * np.log2(self.n)
        return len(self.rows) * (N_CUES + per_dist) + per_dist


class Latent:
    def fit(self, cues, y, seed, steps=1500):
        torch.manual_seed(seed)
        self.n = len(y)
        self.net = torch.nn.Sequential(torch.nn.Linear(N_CUES, K), torch.nn.Tanh(), torch.nn.Linear(K, N_RESP)).double()
        opt = torch.optim.Adam(self.net.parameters(), lr=0.05)
        X, Y = torch.tensor(cues), torch.tensor(y)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(self.net(X), Y).backward()
            opt.step()
        return self

    @torch.no_grad()
    def predict(self, cues, state_shift=None):
        h = self.net[1](self.net[0](torch.tensor(cues)))
        if state_shift is not None:
            h = h + torch.tensor(state_shift)
        return torch.softmax(self.net[2](h), -1).numpy()

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())

    def model_bits(self):
        return self.n_params() * 0.5 * np.log2(self.n)


def steer_latent(model, cues, target=0.1):
    """Shift the k-d state along the direction that most raises p(response 0); scale to reach `target`."""
    W = model.net[2].weight.detach().numpy()
    direction = W[0] - W.mean(0)
    direction /= np.linalg.norm(direction)
    base = model.predict(cues)[:, 0]
    lo, hi = 0.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if (model.predict(cues, mid * direction)[:, 0] - base).mean() < target:
            lo = mid
        else:
            hi = mid
    shift = model.predict(cues, hi * direction)[:, 0] - base
    return {"params_changed": K, "mean_shift_p0": float(shift.mean()), "std_shift_p0": float(shift.std()),
            "frac_situations_shifted_up": float((shift > 0).mean())}


def steer_rules(model, cues, target=0.1):
    """The same global shift for a table: every stored row (and the fallback) must be edited."""
    base = model.predict(cues)[:, 0]
    lam = target / max(1e-9, (1 - base).mean())
    edited = {i: (1 - lam) * r + lam * np.eye(N_RESP)[0] for i, r in model.rows.items()}
    marg = (1 - lam) * model.marg + lam * np.eye(N_RESP)[0]
    new = np.array([edited.get(i, marg) for i in index(cues)])[:, 0]
    shift = new - base
    return {"rows_edited": len(edited), "fallback_edited": 1, "numbers_changed": (len(edited) + 1) * N_RESP,
            "mean_shift_p0": float(shift.mean()), "std_shift_p0": float(shift.std())}


def run(out_dir):
    rows = []
    for kind in ["latent", "rule"]:
        for seed in SEEDS:
            world = World(kind, seed)
            test_x, test_y = world.sample(N_TEST)
            for n in SIZES:
                x, y = world.sample(n)
                rules, latent = Rules().fit(x, y), Latent().fit(x, y, seed)
                marg = (np.bincount(y, minlength=N_RESP) + 0.5) / (n + 0.5 * N_RESP)
                entry = {"world": kind, "seed": seed, "N": n}
                for name, m in [("rules", rules), ("latent", latent)]:
                    entry[name] = {"model_bits": float(m.model_bits()),
                                   "data_bits": float(bits(m.predict(x), y).sum()),
                                   "test_bits_per_sample": float(bits(m.predict(test_x), test_y).mean())}
                    entry[name]["total_bits"] = entry[name]["model_bits"] + entry[name]["data_bits"]
                mb = (N_RESP - 1) * 0.5 * np.log2(n)
                entry["marginal"] = {"model_bits": float(mb), "data_bits": float(bits(np.tile(marg, (n, 1)), y).sum()),
                                     "test_bits_per_sample": float(bits(np.tile(marg, (N_TEST, 1)), test_y).mean())}
                entry["marginal"]["total_bits"] = entry["marginal"]["model_bits"] + entry["marginal"]["data_bits"]
                entry["rules"]["stored_rows"] = len(rules.rows)
                entry["latent"]["n_params"] = latent.n_params()
                if n == SIZES[-1]:
                    entry["steering"] = {"latent": steer_latent(latent, test_x), "rules": steer_rules(rules, test_x)}
                entry["true_entropy_bits_test"] = float(
                    -(world.probs(test_x) * np.log2(np.clip(world.probs(test_x), 1e-12, None))).sum(1).mean())
                rows.append(entry)
                print(f"  idea4 {kind} seed={seed} N={n} done")
    results = {"runs": rows}

    def runs(kind, n=None):
        return [r for r in rows if r["world"] == kind and (n is None or r["N"] == n)]

    p1 = all(r["latent"]["total_bits"] < min(r["rules"]["total_bits"], r["marginal"]["total_bits"])
             and r["latent"]["test_bits_per_sample"] < min(r["rules"]["test_bits_per_sample"], r["marginal"]["test_bits_per_sample"])
             for r in runs("latent") if r["N"] >= 1000)
    p2 = all(r["latent"]["test_bits_per_sample"] >= r["rules"]["test_bits_per_sample"] for r in runs("rule", SIZES[-1]))
    p3 = all(r["steering"]["latent"]["params_changed"] <= 3
             and r["steering"]["rules"]["rows_edited"] == r["rules"]["stored_rows"] for r in runs("latent", SIZES[-1]))
    results["predictions"] = {"P1_latent_shortest_in_latent_world": p1,
                              "P2_latent_not_better_in_rule_world": p2,
                              "P3_steering_edit_counts": p3}
    save_json(out_dir / "idea4_mdl_empathy.json", results)
    plot(out_dir, results)
    return results


def plot(out_dir, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for col, kind in enumerate(["latent", "rule"]):
        for name, style in [("rules", "o-"), ("latent", "s-"), ("marginal", "^--")]:
            ys = [np.mean([x[name]["test_bits_per_sample"] for x in r["runs"] if x["world"] == kind and x["N"] == n]) for n in SIZES]
            ax[col].plot(SIZES, ys, style, label=name)
        ent = np.mean([x["true_entropy_bits_test"] for x in r["runs"] if x["world"] == kind])
        ax[col].axhline(ent, color="gray", lw=0.7, ls=":", label="true entropy")
        ax[col].set_xscale("log")
        ax[col].set_xlabel("training samples N")
        ax[col].set_ylabel("held-out bits / sample")
        ax[col].set_title(f"{kind.upper()} world")
        ax[col].legend()
    for name, style in [("rules", "o-"), ("latent", "s-")]:
        ys = [np.mean([x[name]["total_bits"] for x in r["runs"] if x["world"] == "latent" and x["N"] == n]) / n for n in SIZES]
        ax[2].plot(SIZES, ys, style, label=name)
    ax[2].set_xscale("log")
    ax[2].set_title("LATENT world: total code length / N")
    ax[2].set_ylabel("bits per sample (model + data)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "idea4_mdl_empathy.png", dpi=110)
    plt.close(fig)
