# 09. Instruction Tuning

Instruction tuning is optional for this tutorial. The tiny base model should work first.

## Suggested order

1. Finish TinyStories tokenizer and pretraining.
2. Validate generation and GGUF export.
3. Optionally run Hacker News adaptation.
4. Only then experiment with a small supervised fine-tuning set.

## Why it is optional

- Tiny models can overfit quickly.
- Good instruction datasets add extra preprocessing and evaluation complexity.
- It is easier to reason about failures when the base model pipeline is already stable.

## Reasonable v1 options

- small sampled Alpaca-style data
- hand-curated domain prompts
- short-format response tuning only

Keep this phase narrow and experimental.
