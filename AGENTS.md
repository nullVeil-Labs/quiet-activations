# Project rules

This is a **public** repository. Everything committed is world-readable.

- Never invent facts, citations, results, or completed checks. Cite evidence; label hypotheses and uncertainty. Say "not verified" when appropriate.
- Freeze protocols (`protocols/`) before running an experiment. Edits after a run go in a dated amendment section; never rewrite the original.
- Preserve raw results, report failures and nulls, and verify units and methods before comparisons.
- Toy models show what is *possible*, not what real LLMs *do*. Geometry, compression, or steering results alone do not establish feelings, empathy, selfhood, or consciousness. Say which kind of evidence each result is.
- Minimize dependencies; prefer the pinned runtime in `requirements.txt`. Before installing/updating, verify official provenance, exact versions, the full dependency tree, current vulnerability advisories, package integrity, and release age of at least 14 days. Record the evidence in `DEPENDENCIES.md`.
- Do not execute unreviewed downloaded code or bypass unresolved security checks. Never promise software is risk-free.
- Do not hash model files.
- Never commit model weights, `.venv/`, secrets, tokens, absolute local paths, or personal data. Models live in `models/` (gitignored); the README says where to get them.
- Keep one clear entry point (`run.py`) and small, plainly named modules. Avoid duplicate implementations and unnecessary folders or abstractions.
- Separate code (`nightshift/`), protocols (`protocols/`), raw results (`results/<run_id>/`), and analysis (`ANALYSIS.md`). Use unique run IDs; never overwrite results.
- Keep README current with canonical paths, project status, and the next step.
