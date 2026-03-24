# 05. Pretraining On TinyStories

This is the main tutorial path. Get this working before touching Hacker News.

## Sanity run first

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/train_pretrain.py \
  --model-config configs/model/cap_26m.json \
  --train-config configs/train/tinystories_sanity.json
```

The sanity config uses a reduced token budget so you can confirm:

- device selection works
- loss decreases
- checkpoints save correctly
- evaluation can read the saved output

## Full tutorial run

```bash
python scripts/train_pretrain.py \
  --model-config configs/model/cap_26m.json \
  --train-config configs/train/tinystories_full.json
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
