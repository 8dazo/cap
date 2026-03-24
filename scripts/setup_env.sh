#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first, for example with: brew install uv"
  exit 1
fi

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .

mkdir -p artifacts/.hf/hub artifacts/.hf/datasets artifacts/tokenizers artifacts/checkpoints artifacts/exports

cat <<'EOF'
Environment created.

Recommended next commands:
  source .venv/bin/activate
  export HF_HOME="$PWD/artifacts/.hf"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  python scripts/check_backend.py
EOF
