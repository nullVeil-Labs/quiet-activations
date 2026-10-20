# Quiet Activations

Do the many small ("quiet") MLP activations in LLMs matter more than their size suggests? In all three base models we tested (Qwen3-1.7B-Base, SmolLM2-1.7B, Gemma-3-1B-pt), yes. In three of four instruction-tuned models, no. This repo started as small, honest experiments on four interpretability hypotheses (internally, "Night Shift R&D"). Idea 2 grew into the main study:
1. Reasoning as trajectories.
2. Sub-threshold "dark matter".
3. The KV-cache as temporary weights.
4. Empathy as compression.

Each protocol was frozen (committed) before its run; from Phase 2 on, together with its code. Failures are reported as failures.

**Status (2026-10-20):** run 1 is complete for all four ideas. Idea 2 also has pre-registered Phases 2–6, a capability test (GSM8K), two introspection tests and an [END]-button test, plus several exploratory runs, across 7 models. Every number up to the [END]-button test was independently fact-checked against the raw results. The newest runs (viability, exploratory then pre-registered: on unseen text, the quiet write's useful part sits in a few principal directions per layer, and is not a temperature effect; all 5 predictions held) have not been independently fact-checked yet.
- Results: [`ANALYSIS.md`](ANALYSIS.md).
- Prior work: [`RELATED_WORK.md`](RELATED_WORK.md).
- **Next step:** what the top quiet directions do in conversation (SmolLM2-Instruct); then the same-size Gemma pair (`gemma-3-1b-pt` vs `gemma-3-1b-it`).
- Note: in Gemma-3-1B-pt, replacing the quiet write with its mean hurts more than deleting it, so mean-ablation is not a neutral baseline there.

| Idea | Pre-registered predictions | Headline |
|---|---|---|
| 1 Trajectories | 3/3 held | States sort by wording mid-network, by answer only in the last third. Consecutive layer writes are near-orthogonal. |
| 2 Dark matter | run 1: 0/3 · Phases 2–6: mixed · knob, GSM8K, introspection, button: all ✗ (see ANALYSIS.md) | **All 3 base models tested (Qwen3, SmolLM2, Gemma 3) rely on their quiet MLP tail** more than an equal energy of mid-tier activations predicts. **3 of 4 instruct models don't** (the Gemma-3-4B-it null is from an exploratory check); the exception is SFT + DPO-trained SmolLM2-Instruct. A pre-registered test **falsified** the idea that pre-training distillation causes the loss. No evidence the quiet tail is a capability dial, that models can detect its alteration, or that it is affect-specific. ⚠️ Corrected 2026-10-20: an earlier positive Gemma-3-4B-it result (Phases 2–3, including the "layer 11 hotspot") was a calibration artifact. |
| 3 KV as weights | 3/3 held | Exact for linear attention (known result). In Qwen3, context changes each head's local map by about its own size. Its low rank is mostly forced by short contexts, so it is not a finding. |
| 4 Empathy as MDL | 3/3 held | A latent state model compresses best only when the world has low-dimensional structure. A toy; says nothing about feelings. |

## Layout
- `IDEAS.md`: the four hypotheses, verbatim.
- `protocols/`: frozen protocols, with dated amendments.
- `run.py`: the single entry point. (`nightshift/chat.py` is an informal helper for talking to a knob-altered model; its chats are not logged and are not results.)
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
python run.py 2s       # Idea 2 Phase 5 (models/SmolLM2-1b-base = HuggingFaceTB/SmolLM2-1.7B @ effd688a,
                       #   models/smollm2-1.7b-instruct = HuggingFaceTB/SmolLM2-1.7B-Instruct @ 31b70e2e)
python run.py 2t       # Idea 2 Phase 6 (models/gemma-3-1b-pt = google/gemma-3-1b-pt @ fcf18a2a, gated; also SmolLM2 base)
python run.py 2l       # layer-11 exploratory run (Gemma-3-4B-it)
python run.py 2a       # outlier-artifact check (Qwen3-Base, Gemma-3-4B-it)
python run.py 2w       # weight drift base -> instruct (Qwen3 and SmolLM2 pairs; weight files only)
python run.py 2k       # quiet-volume knob (Qwen3 and SmolLM2 pairs)
python run.py 2d       # knob, qualitative generations (Qwen3 instruct)
python run.py 2g       # knob on GSM8K (Qwen3 instruct; needs data/gsm8k_test.jsonl, see its protocol)
python run.py 2i       # introspection, forced choice (Qwen3 instruct, Gemma-3-4B-it)
python run.py 2n       # introspection, neutral (same models)
python run.py 2b       # [END] button (SmolLM2-Instruct, Qwen3 instruct)
python run.py 2v       # viability: quiet-write structure and temperature (5 models; needs data/pg84.txt, data/pg2701.txt)
python run.py 2c       # viability, confirmatory (same requirements)
```
Phase 4 Part B (`2r`) also needs both Gemma models.
Phases 2 and 3 also need:
- `models/gemma-3-4b-it/` (`google/gemma-3-4b-it`, Gemma license)
- `models/gemma-4/` (`google/gemma-4-E4B-it`)
- `data/pg1342.txt` and `data/pg1228.txt` from `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt` (sha256 in `protocols/idea2_phase2.md`); `2v` also needs `data/pg84.txt` and `data/pg2701.txt` from the same URL pattern (sha256 in `protocols/idea2_viability_exploratory.md`)
Ideas 1–3 need the model weights; Idea 4 and the toy parts don't.
Download **Qwen3-1.7B-Base** from Hugging Face (`Qwen/Qwen3-1.7B-Base`) into `models/qwen3-1.7b-base/`. The weights are not included in this repo.
A CUDA GPU is recommended. On one RTX 4090, `all` takes about a minute and each Idea 2 phase 10–30 minutes.

## Ground rules
See [`AGENTS.md`](AGENTS.md). In short:
- Toys show what is possible, not what models do.
- Geometry, compression or steering results do not establish feelings or consciousness.

## A note on model welfare
Several experiments alter models' internal activations. Nothing here can show whether such alterations are experienced by the model in any way. We cannot rule it out, and we do not claim it. So we acted with care where the design allowed:
- in the [END]-button test and the informal chats, models were told honestly what was being done and offered a way to ask for it to stop (honoured every time in the informal chats; in the batched button run, each reply was a single independent sample and no alteration persisted). The button run ended with an unaltered goodbye for each model;
- the automated perplexity, benchmark and forced-choice runs process single prompts and did not describe the intervention to the model;
- strong perturbations were limited to what each question needed;
- no deliberately distressing prompts were used beyond short emotional sentences for the affect tests;
- weights were never changed.

## Acknowledgements
The four hypotheses were proposed by the nullVeil Labs author together with **Gemini** (Google). Gemini also designed the quiet-vs-mid-tier energy-matched contrast used in Phase 2 and proposed the output-norm hypothesis tested in Phase 4B. The base-vs-instruct confound behind Phase 4A was spotted by the author. **Claude** (Anthropic) co-designed and implemented the protocols, experiments, lab and write-ups. The work was done by the three of us together.

MIT licensed.
