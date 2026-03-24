# 02. Environment Setup

This project standardizes on `uv + venv` for Python environment management on macOS.

## System tools

Install or verify:

- Xcode command line tools
- `uv`
- `git`
- `cmake` if you plan to build `llama.cpp`

Example:

```bash
xcode-select --install
brew install uv cmake git
```

## Create the environment

From the repo root:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

If PyTorch installation on your Mac requires a newer wheel path than the one resolved by `uv pip install -e .`, install PyTorch first, then rerun the editable install:

```bash
uv pip install torch torchvision torchaudio
uv pip install -e .
```

## Cache layout

Keep Hugging Face cache content inside the repo so disk usage is visible and easy to clean up:

```bash
export HF_HOME="$PWD/artifacts/.hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
```

## MPS safety knobs

Use CPU fallback for unsupported MPS ops:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Only touch the high watermark limit if you understand the memory risk:

```bash
# Example only. Do not enable by default.
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## Verify backend

Run:

```bash
python scripts/check_backend.py
```

Expected result:

- `mps_available: True` on a supported Apple silicon setup
- selected device resolves to `mps`

If you land on `cpu`, finish the tutorial in CPU mode only for smoke tests or fix the PyTorch/macOS setup first.
