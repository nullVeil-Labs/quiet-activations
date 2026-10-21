# Dependency evidence

Checked 2026-10-19. `requirements.txt` pins the full tree (44 packages), and the installed environment matches it exactly (`pip freeze`), with `pip check` reporting no broken requirements.

- **Provenance:** PyPI, plus the official PyTorch index `download.pytorch.org/whl/cu130` for `torch==2.13.0+cu130`. Python 3.13.15.
- **Direct dependencies:** torch 2.13.0+cu130, transformers 5.16.1, numpy 2.5.2, matplotlib 3.11.1.
- **Release age (≥ 14 days):** the first PyPI upload date of every pinned version is on or before 2026-10-02. A first unpinned install pulled 7 transitive packages released after 2026-10-05. They were pinned back to older versions: certifi 2026.7.22, filelock 3.32.4, fonttools 4.63.0, hf-xet 1.6.0, pillow 12.3.0, tokenizers 0.23.1, typer 0.27.2.
- **Advisories:** all 44 pinned versions were queried against the OSV database (`api.osv.dev/v1/querybatch`, ecosystem PyPI). No advisories were returned.
- **Integrity:** pip verifies the sha256 hashes that the indexes publish for downloaded files. The lock does not use `--require-hashes`, so not verified beyond that.
- **Not verified:** a manual review of package source code. No software is promised risk-free.

**Models** (not part of this repository; not hashed, by policy; file sizes were checked against Hugging Face for the downloaded ones):
- Qwen3-1.7B-Base (`Qwen/Qwen3-1.7B-Base`), Apache-2.0. Local copy; revision not recorded.
- Qwen3-1.7B (`Qwen/Qwen3-1.7B` @ `70d244cc`), Apache-2.0.
- SmolLM2-1.7B (`HuggingFaceTB/SmolLM2-1.7B` @ `effd688a`) and SmolLM2-1.7B-Instruct (`@ 31b70e2e`), Apache-2.0.
- Gemma-3-1B-pt (`google/gemma-3-1b-pt` @ `fcf18a2a`) and Gemma-3-4B-it (`google/gemma-3-4b-it`, local copy; revision not recorded), Gemma Terms of Use.
- Gemma-4-E4B-it (`google/gemma-4-E4B-it`, local copy; revision not recorded), Apache-2.0 per its model card.

**Data** (not committed; sources and sha256 in the protocols): Project Gutenberg #1342 and #1228 (`protocols/idea2_phase2.md`); GSM8K test set, MIT (`protocols/idea2_gsm8k_knob.md`); SimpleQA, MIT, and TruthfulQA MC, Apache-2.0 (`protocols/idea2_hallucination_knob.md`).
