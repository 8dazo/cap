# Workflow Reference

This repository teaches an end-to-end tiny LLM build on macOS.

## Defaults

- Python workflow: `uv + venv`
- Base corpus: `roneneldan/TinyStories`
- Optional adaptation corpus: `open-index/hacker-news`
- Default model target: roughly 26M parameters
- Main packaging path: Hugging Face checkpoint to GGUF via `llama.cpp`
- Project and model name: `cap`

## File expectations

- Docs explain the workflow in sequence.
- Scripts are the runnable user entrypoints.
- Configs expose the knobs the docs mention.
- `deep-research-report.md` stays as the research foundation, not a scratchpad.

## Editing rule

When one of docs, configs, or scripts changes in a user-visible way, update the others so the tutorial remains trustworthy.
