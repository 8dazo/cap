# 05. Pretraining On TinyStories

This is the main tutorial path. Get this working before touching Hacker News.

## Sanity run first

```bash
source .venv/bin/activate
export HF_TOKEN="your_token_here"
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/train_pretrain.py \
  --model-config configs/model/cap_26m.json \
  --train-config configs/train/tinystories_sanity.json
```

The sanity config uses a reduced token budget so you can confirm:

- device selection works
- config validation passes before the job starts
- startup stage logs show tokenizer loading, model build, dataloader setup, and the first batch
- loss decreases
- checkpoints save correctly
- evaluation can read the saved output

## Full tutorial run

```bash
python scripts/train_pretrain.py \
  --model-config configs/model/cap_26m.json \
  --train-config configs/train/tinystories_full.json
```

## Fast-dev run

Use this when you want a quick iteration cycle after changing code or configs:

```bash
python scripts/train_pretrain.py \
  --model-config configs/model/cap_26m.json \
  --train-config configs/train/tinystories_fast_dev.json
```

## Quality-oriented run

Use this only in your own terminal, because it is a longer-running job:

```bash
python scripts/train_pretrain.py \
  --model-config configs/model/cap_26m.json \
  --train-config configs/train/tinystories_quality.json
```

## Quality continuation run

Use this when `cap-26m-fast-dev` is your strongest checkpoint and you want a safer long run than restarting from scratch:

```bash
python scripts/train_pretrain.py \
  --model-config configs/model/cap_26m.json \
  --train-config configs/train/tinystories_quality_continue.json \
  --init-checkpoint artifacts/checkpoints/cap-26m-fast-dev
```

## Model shape

The default model target is approximately 26M parameters with:

- 8 layers
- hidden size 512
- 8 attention heads
- 2 key-value heads
- intermediate size 1408
- RoPE and RMSNorm through the Hugging Face LLaMA configuration path

## Mac-first notes

- MPS uses a single device.
- Float16 autocast is the practical default on Mac.
- Gradient checkpointing is optional and enabled in the fuller config.
- Keep sequence length at 256 while validating the end-to-end path.
- The training script prints a run summary before it begins token processing so you can catch wrong paths or budgets early.
- The `fast-dev` config is for quick debugging; the `quality` config is for a longer run you should launch in your own terminal.
- If the scratch `quality` run degrades model quality, prefer `tinystories_quality_continue.json` with `--init-checkpoint artifacts/checkpoints/cap-26m-fast-dev`.
