# Quiet Activations

Do the many small ("quiet") MLP activations in LLMs matter more than their size suggests? In all three base models we tested (Qwen3-1.7B-Base, SmolLM2-1.7B, Gemma-3-1B-pt), yes. In three of four instruction-tuned models, no. This repo started as small, honest experiments on four interpretability hypotheses (internally, "Night Shift R&D"). Idea 2 grew into the main study:
1. Reasoning as trajectories.
2. Sub-threshold "dark matter".
3. The KV-cache as temporary weights.
4. Empathy as compression.

Each protocol was frozen (committed) before its run. Failures are reported as failures.

**Status (2026-10-19):** run 1 is complete for all four ideas. Idea 2 also has an exploratory follow-up and two pre-registered phases on 3 models.
- Results: [`ANALYSIS.md`](ANALYSIS.md).
- Prior work: [`RELATED_WORK.md`](RELATED_WORK.md).
- **Next step:** the same-size Gemma pair (`gemma-3-1b-pt` vs `gemma-3-1b-it`), and the quiet-volume knob.

| Idea | Pre-registered predictions | Headline |
|---|---|---|
| 1 Trajectories | 3/3 held | States sort by wording mid-network, by answer only in the last third. Consecutive layer writes are near-orthogonal. |
| 2 Dark matter | run 1 → Phase 6 (see ANALYSIS.md) | **All 3 base models tested (Qwen3, SmolLM2, Gemma 3) rely on their quiet MLP tail** more than an equal energy of mid-tier activations predicts. **3 of 4 instruct models don't**, the exception being lightly tuned SmolLM2-Instruct. A pre-registered test **falsified** the idea that distillation causes the loss. ⚠️ Corrected 2026-10-20: an earlier Gemma 3 "layer 11" result was a calibration artifact. No evidence the signal is affect-specific. |
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
python run.py 2r       # Idea 2 Phase 4 (also needs models/qwen3-1.7b/ = Qwen/Qwen3-1.7B @ 70d244cc)
python run.py 2t       # Idea 2 Phase 6 (models/gemma-3-1b-pt = google/gemma-3-1b-pt @ fcf18a2a, gated)
python run.py 2s       # Idea 2 Phase 5 (models/SmolLM2-1b-base = HuggingFaceTB/SmolLM2-1.7B @ effd688a,
                       #   models/smollm2-1.7b-instruct = HuggingFaceTB/SmolLM2-1.7B-Instruct @ 31b70e2e)
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

## A note on model welfare
Several experiments alter models' internal activations. Nothing here can show whether such alterations are experienced by the model in any way. We cannot rule it out, and we do not claim it. So we acted with care:
- models were told honestly what was being done, and offered a way to ask for it to stop, which was always honoured;
- strong perturbations were limited to what each question needed;
- no distressing content was used;
- weights were never changed;
- each session ended with an unaltered goodbye.

## Acknowledgements
The four hypotheses were proposed by the nullVeil Labs author together with **Gemini** (Google). Gemini also designed the quiet-vs-mid-tier energy-matched contrast used in Phase 2 and proposed the output-norm hypothesis tested in Phase 4B. The base-vs-instruct confound behind Phase 4A was spotted by the author. **Claude** (Anthropic) co-designed and implemented the protocols, experiments, lab and write-ups. The work was done by the three of us together.

MIT licensed.
