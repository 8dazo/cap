# 01. Overview And Prereqs

This project teaches how to build a tiny local LLM on a Mac in a way that is realistic for 24 GB unified memory and modest free disk. The learning goal is reproducibility, not state-of-the-art quality.

Read [`deep-research-report.md`](../deep-research-report.md) for the full research background. This tutorial distills it into a guided implementation path.

## What you will build

- A ByteLevel BPE tokenizer trained on TinyStories.
- A roughly 26M parameter decoder-only Transformer using a LLaMA-style configuration.
- A Mac-first PyTorch training workflow that targets MPS and falls back safely.
- A Hugging Face compatible checkpoint that can be exported to GGUF and run with `llama.cpp`.
- An optional second-stage Hacker News domain-adaptation run.

## Recommended Mac baseline

- Apple silicon Mac strongly preferred.
- macOS 12.3 or newer for MPS support.
- 24 GB unified memory is a practical baseline for the tutorial defaults.
- 30-40 GB free disk keeps caches, environments, checkpoints, and optional builds manageable.

## What gets downloaded

- Python package environment via `uv` and `venv`.
- PyTorch and the Hugging Face stack.
- TinyStories streamed from Hugging Face by default.
- Optional `llama.cpp` source build for GGUF export and local inference.
- Optional Hacker News streamed dataset for the second track.

## Important constraints

- MPS is single-device. Do not design this tutorial around distributed training.
- Unsupported operations can fall back to CPU. Performance may vary by macOS and PyTorch version.
- Tiny models can produce coherent text, but they will not behave like large frontier assistants.
- Start with short sanity runs before attempting larger token budgets.

## Tutorial order

1. [`02-environment-setup.md`](02-environment-setup.md)
2. [`03-project-layout-and-guidelines.md`](03-project-layout-and-guidelines.md)
3. [`04-tokenizer-training.md`](04-tokenizer-training.md)
4. [`05-pretraining-on-tinystories.md`](05-pretraining-on-tinystories.md)
5. [`06-evaluation-and-sampling.md`](06-evaluation-and-sampling.md)
6. [`07-export-to-gguf.md`](07-export-to-gguf.md)
7. [`08-hacker-news-domain-adaptation.md`](08-hacker-news-domain-adaptation.md)
8. [`09-instruction-tuning.md`](09-instruction-tuning.md)

## Before you begin

- Keep the project root writable and avoid moving artifact directories mid-run.
- Do not add a `.ai/` folder to this repo.
- Treat TinyStories as the required first path; Hacker News is optional and intentionally later.
