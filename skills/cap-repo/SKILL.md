---
name: cap-repo
description: Use this skill when working in this repository on the Cap macOS tiny LLM tutorial, including setup, TinyStories pretraining, Hacker News adaptation, GGUF export, and the GitHub issue to PR to merge pipeline for the cap repo.
---

# Cap Repo

Use this skill for tasks in this repository that touch the tutorial workflow or the GitHub delivery workflow.

## Core workflow

1. Read `docs/01-overview-and-prereqs.md` before changing build steps.
2. Keep `TinyStories` as the default base dataset and `open-index/hacker-news` as an optional second-stage adaptation track.
3. Ensure any command shown in `docs/` maps to a real file under `scripts/`, `configs/`, or `src/`.
4. Keep all local outputs under `artifacts/`.
5. Do not create a `.ai/` folder in this repository.
6. Keep the project and model name as `cap` in docs, configs, checkpoints, tokenizer outputs, and exports.

## GitHub pipeline

Follow this repo workflow unless the user explicitly asks otherwise:

1. Create or update a GitHub issue for the work.
2. Implement changes on a non-`main` branch.
3. Open a pull request that links the issue.
4. Merge the pull request into `main` after checks and review are satisfied.
5. Keep the remote aligned with `https://github.com/8dazo/cap`.

## When editing

- Preserve the Mac-first assumptions and MPS caveats.
- Prefer small sanity runs before larger defaults.
- Keep the Hugging Face checkpoint path intact because GGUF export depends on it.
- If dataset behavior or filtering changes, update the relevant tutorial docs in the same change.
- If the GitHub workflow changes, update `references/github.md`.

## References

- For operating notes and repo conventions, read `references/workflow.md`.
- For MPS-specific cautions, read `references/mps.md`.
- For the issue to PR to merge flow, read `references/github.md`.
