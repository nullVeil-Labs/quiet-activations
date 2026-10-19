# Night Shift R&D

Small, honest experiments on four interpretability hypotheses about LLMs:
1. Reasoning as trajectories.
2. Sub-threshold "dark matter".
3. The KV-cache as temporary weights.
4. Empathy as compression.

Each protocol was frozen (committed) before its run. Failures are reported as failures.

**Status (2026-10-19):** run 1 is complete for all four ideas. Idea 2 also has an exploratory follow-up and two pre-registered phases on 3 models.
- Results: [`ANALYSIS.md`](ANALYSIS.md).
- Prior work: [`RELATED_WORK.md`](RELATED_WORK.md).
- **Next step:** find what Gemma 3's layer-11 quiet activations encode, and test causally why Gemma 4 differs.

| Idea | Pre-registered predictions | Headline |
|---|---|---|
| 1 Trajectories | 3/3 held | States sort by wording mid-network, by answer only in the last third. Consecutive layer writes are near-orthogonal. |
| 2 Dark matter | run 1: 0/3 · Phase 2: 2/4 · Phase 3: 2/5 | **Model-specific.** Quiet MLP activations matter more than an equal energy of mid-tier ones in Qwen3 (diffusely, ~2×) and Gemma 3 (+0.21 nats, mostly from one layer, layer 11). The reverse holds in Gemma 4. No evidence they specifically carry affect. |
| 3 KV as weights | 3/3 held | Exact for linear attention (known result). In Qwen3, context changes each head's local map by about its own size. Its low rank is mostly forced by short contexts, so it is not a finding. |
| 4 Empathy as MDL | 3/3 held | A latent state model compresses best only when the world has low-dimensional structure. A toy; says nothing about feelings. |

## Layout
- `IDEAS.md`: the four hypotheses, verbatim.
- `protocols/`: frozen protocols, with dated amendments.
- `run.py`: the single entry point.
- `nightshift/`: one module per idea.
- `results/<run_id>/`: raw JSON and figures. Never overwritten.
- `lab/index.html`: an interactive lab. Open it in a browser; it reads `lab/data.js`.
- `DEPENDENCIES.md`: provenance, release-age and advisory checks.

## Run it
```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
python run.py all      # ideas 1-4, refreshes lab/data.js
python run.py 2f       # exploratory Idea 2 follow-up
python run.py 2p       # Idea 2 Phase 2 (needs the Gemma models and data/, see below)
python run.py 2q       # Idea 2 Phase 3 (same requirements)
```
Phases 2 and 3 also need:
- `models/gemma-3-4b-it/` (`google/gemma-3-4b-it`, Gemma license)
- `models/gemma-4/` (`google/gemma-4-E4B-it`)
- `data/pg1342.txt` and `data/pg1228.txt` from `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt` (sha256 in `protocols/idea2_phase2.md`)
Ideas 1–3 need the model weights; Idea 4 and the toy parts don't.
Download **Qwen3-1.7B-Base** from Hugging Face (`Qwen/Qwen3-1.7B-Base`) into `models/qwen3-1.7b-base/`. The weights are not included in this repo.
A CUDA GPU is recommended; one RTX 4090 runs everything in a few minutes.

## Ground rules
See [`AGENTS.md`](AGENTS.md). In short:
- Toys show what is possible, not what models do.
- Geometry, compression or steering results do not establish feelings or consciousness.

MIT licensed.
