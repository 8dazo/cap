# 04. Tokenizer Training

The tutorial uses a ByteLevel BPE tokenizer inspired by the MiniMind workflow in the research report. TinyStories is the default source because it is compact, coherent, and well-suited to tiny-language-model experiments.

## Recommended first run

```bash
source .venv/bin/activate
export HF_HOME="$PWD/artifacts/.hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
python scripts/train_tokenizer.py \
  --config configs/model/cap_26m.json \
  --output-dir artifacts/tokenizers/cap-bytebpe \
  --max-stories 200000
```

## What the script does

- Streams TinyStories from Hugging Face
- Shuffles a bounded window for a reproducible sample
- Trains a ByteLevel BPE tokenizer
- Saves Hugging Face compatible tokenizer artifacts
- Stores a small metadata file with the training settings

## Defaults worth keeping

- Vocabulary size: 6400
- Special tokens:
  - `<|endoftext|>`
  - `<|im_start|>`
  - `<|im_end|>`

## Success criteria

- `tokenizer.json` exists in the output directory
- `tokenizer_config.json` and special token metadata exist
- The tokenizer can be loaded by `AutoTokenizer.from_pretrained(...)`
