# 06. Evaluation And Sampling

Use a saved checkpoint to test that the model can generate text and that the tokenizer/checkpoint layout is valid.

## Run structured evaluation

```bash
source .venv/bin/activate
export HF_HOME="$PWD/artifacts/.hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_TOKEN="your_token_here"
python scripts/eval_checkpoint.py \
  --checkpoint artifacts/checkpoints/cap-26m-sanity \
  --sequence-length 256 \
  --batch-size 8 \
  --eval-batches 20
```

This reports:

- average loss across a bounded TinyStories stream
- perplexity derived from that average loss
- per-batch progress so evaluation does not look stalled

## Run a quick sample

```bash
source .venv/bin/activate
python scripts/eval_sample.py \
  --checkpoint artifacts/checkpoints/cap-26m-sanity \
  --prompt "Once upon a time," \
  --max-new-tokens 120
```

## What to look for

- The checkpoint loads without missing tokenizer files.
- Generation runs on `mps` or `cpu`.
- Output is rough but coherent enough to prove the training path works.
- Structured eval should complete without checkpoint-loading errors and produce stable loss and perplexity numbers you can compare across runs.

## Optional evaluation ideas

- Hold-out TinyStories loss tracking
- WikiText-2 perplexity
- fixed prompt comparisons across checkpoints

This tutorial keeps the default evaluation lightweight so local iteration stays fast.

## Speed vs Quality

If you want faster iteration:

- use the sanity config before the full config
- use `configs/train/tinystories_fast_dev.json` after code changes
- keep `sequence_length=256`
- reduce `eval_batches` during debugging
- keep `HF_TOKEN` exported so dataset access is less rate-limited
- use sample generation sparingly during training and rely on bounded eval first

If you want better quality:

- train longer than the sanity token budget
- use `configs/train/tinystories_quality.json` or another longer TinyStories config after sanity succeeds
- compare eval loss and perplexity across checkpoints instead of relying only on one prompt
- add a second-stage adaptation pass only after the base TinyStories model is stable
