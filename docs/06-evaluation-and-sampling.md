# 06. Evaluation And Sampling

Use a saved checkpoint to test that the model can generate text and that the tokenizer/checkpoint layout is valid.

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

## Optional evaluation ideas

- Hold-out TinyStories loss tracking
- WikiText-2 perplexity
- fixed prompt comparisons across checkpoints

This tutorial keeps the default evaluation lightweight so local iteration stays fast.
