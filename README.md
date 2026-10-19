# Night Shift R&D

Small, honest experiments on four interpretability hypotheses about LLMs:
1. Reasoning as trajectories.
2. Sub-threshold "dark matter".
3. The KV-cache as temporary weights.
4. Empathy as compression.

Each protocol was frozen (committed) before its run. Failures are reported as failures.

**Status (2026-10-19):** run 1 is complete for all four ideas, plus an exploratory follow-up and a pre-registered Phase 2 replication of Idea 2 on 3 models.
- Results: [`ANALYSIS.md`](ANALYSIS.md).
- Prior work: [`RELATED_WORK.md`](RELATED_WORK.md).
- **Next step:** a Phase 3 Idea 2 protocol with a difference-based criterion, per-chunk raw results, and a look at why Gemma 4 shows no effect.

| Idea | Pre-registered predictions | Headline |
|---|---|---|
| 1 Trajectories | 3/3 held | States sort by wording mid-network, by answer only in the last third. Consecutive layer writes are near-orthogonal. |
| 2 Dark matter | 0/3 held, then Phase 2: 2/4 held | **Not replicated as registered.** Model-dependent: in Qwen3, quiet activations hurt ~2× more than energy-matched controls (CI excludes 1); in Gemma 3, they hurt while the controls do nothing; in Gemma 4, no effect. |
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
```
Ideas 1–3 need the model weights; Idea 4 and the toy parts don't.
Download **Qwen3-1.7B-Base** from Hugging Face (`Qwen/Qwen3-1.7B-Base`) into `models/qwen3-1.7b-base/`. The weights are not included in this repo.
A CUDA GPU is recommended; one RTX 4090 runs everything in a few minutes.

## Ground rules
See [`AGENTS.md`](AGENTS.md). In short:
- Toys show what is possible, not what models do.
- Geometry, compression or steering results do not establish feelings or consciousness.

MIT licensed.
