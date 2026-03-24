# 08. Hacker News Domain Adaptation

This track is optional and comes after TinyStories pretraining. The dataset is larger and noisier, so it is not the base tutorial corpus.

## Why use it

Hacker News can help the model pick up:

- short technical discussion style
- startup and engineering vocabulary
- headline and comment patterns

It is better suited for domain adaptation than for the first clean end-to-end run.

## Dataset strategy

- stream `open-index/hacker-news`
- avoid full local materialization by default
- filter out deleted and dead items
- keep only text-bearing rows
- separate story titles from comments
- cap the number of examples or tokens for the Mac-friendly run

## Preview filtered text

```bash
source .venv/bin/activate
python scripts/prepare_hackernews.py \
  --config configs/train/hackernews_adapt.json \
  --preview 10
```

## Run a small adaptation job

```bash
python scripts/adapt_hackernews.py \
  --model-config configs/model/cap_26m.json \
  --train-config configs/train/hackernews_adapt.json \
  --checkpoint artifacts/checkpoints/cap-26m-full
```

## Recommended guardrails

- keep adaptation token budgets small at first
- prefer recent year slices over the full archive
- evaluate whether the base model gets more tech-flavored without collapsing story quality completely
