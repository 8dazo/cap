# 03. Project Layout And Guidelines

This repo is intentionally small and tutorial-shaped. The code and docs should stay synchronized.

## Layout

- `docs/`: tutorial sequence and operational notes
- `scripts/`: user-facing entrypoints for setup, training, evaluation, and data preparation
- `src/cap/`: shared Python utilities
- `configs/`: model and training configs
- `artifacts/`: local outputs, caches, tokenizers, checkpoints, and exports
- `skills/`: repo-local agent guidance
- `deep-research-report.md`: detailed research source document

## Rules for this repo

- Keep all tutorial commands runnable from repo root.
- Prefer streamed datasets over full downloads.
- Start with sanity configs before full configs.
- Fail fast on invalid config values before starting long runs.
- Keep tokenizer, checkpoints, and exports under `artifacts/`.
- Avoid hidden magic in docs. Every command should point to a real file in this repo.
- Do not create a `.ai/` folder.

## Naming conventions

- Tokenizer output: `artifacts/tokenizers/<name>`
- Checkpoints: `artifacts/checkpoints/<run-name>`
- Exports: `artifacts/exports/<run-name>`
- Configs: `configs/model/*.json` and `configs/train/*.json`

## Standard entrypoints

- `python scripts/check_backend.py`
- `python scripts/train_tokenizer.py --config configs/model/cap_26m.json`
- `python scripts/train_pretrain.py --model-config ... --train-config ...`
- `python scripts/eval_sample.py --checkpoint ...`
- `python scripts/prepare_hackernews.py --config configs/train/hackernews_adapt.json`
- `python scripts/adapt_hackernews.py --model-config ... --train-config ... --checkpoint ...`

## Working style

- Make one small run succeed before scaling tokens or context.
- Save frequent checkpoints during long local runs.
- Treat Hacker News as a deliberate optional branch, not the default base data.

## GitHub workflow

- Keep the local repo connected to `https://github.com/8dazo/cap`.
- Open or update a GitHub issue before substantial implementation work.
- Do feature work on a non-`main` branch.
- Open a pull request that links the issue.
- Merge to `main` only after the pull request is ready.
