#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: scripts/export_gguf.sh <checkpoint-dir> [output-prefix]"
  exit 1
fi

CHECKPOINT_DIR="$1"
OUTPUT_PREFIX="${2:-}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory not found: $CHECKPOINT_DIR"
  exit 1
fi

if [[ ! -d "artifacts/llama.cpp" ]]; then
  echo "artifacts/llama.cpp not found. Build llama.cpp first."
  echo "Run:"
  echo "  git clone https://github.com/ggml-org/llama.cpp artifacts/llama.cpp"
  echo "  cmake -S artifacts/llama.cpp -B artifacts/llama.cpp/build"
  echo "  cmake --build artifacts/llama.cpp/build --config Release"
  exit 1
fi

mkdir -p artifacts/exports

CHECKPOINT_NAME="$(basename "$CHECKPOINT_DIR")"
if [[ -z "$OUTPUT_PREFIX" ]]; then
  OUTPUT_PREFIX="artifacts/exports/${CHECKPOINT_NAME}"
fi

F16_OUT="${OUTPUT_PREFIX}-f16.gguf"
Q4_OUT="${OUTPUT_PREFIX}-q4_0.gguf"

python artifacts/llama.cpp/convert_hf_to_gguf.py \
  "$CHECKPOINT_DIR" \
  --outfile "$F16_OUT" \
  --outtype f16

artifacts/llama.cpp/build/bin/llama-quantize \
  "$F16_OUT" \
  "$Q4_OUT" \
  Q4_0

echo "Export complete:"
echo "  $F16_OUT"
echo "  $Q4_OUT"
