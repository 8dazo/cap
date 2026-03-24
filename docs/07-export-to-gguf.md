# 07. Export To GGUF

The main deployment path for this tutorial is Hugging Face checkpoint to GGUF, then local inference with `llama.cpp`.

## Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp artifacts/llama.cpp
cmake -S artifacts/llama.cpp -B artifacts/llama.cpp/build
cmake --build artifacts/llama.cpp/build --config Release
```

## Convert a checkpoint

```bash
python artifacts/llama.cpp/convert_hf_to_gguf.py \
  artifacts/checkpoints/cap-26m-sanity \
  --outfile artifacts/exports/cap-26m-sanity-f16.gguf \
  --outtype f16
```

## Quantize

```bash
artifacts/llama.cpp/build/bin/llama-quantize \
  artifacts/exports/cap-26m-sanity-f16.gguf \
  artifacts/exports/cap-26m-sanity-q4_0.gguf \
  Q4_0
```

## Run

```bash
artifacts/llama.cpp/build/bin/llama-cli \
  -m artifacts/exports/cap-26m-sanity-q4_0.gguf \
  -p "Once upon a time," \
  -n 120
```

## Notes

- Keep the Hugging Face checkpoint format intact before export.
- Use the same tokenizer artifacts that were saved with the checkpoint.
- Treat GGUF export as the packaging step after the base training path works.
