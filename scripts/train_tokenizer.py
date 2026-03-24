#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from cap.config import ConfigError, load_json_config, print_run_summary, validate_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TinyStories ByteLevel BPE tokenizer.")
    parser.add_argument("--config", required=True, help="Path to model config JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to save tokenizer artifacts.")
    parser.add_argument("--max-stories", type=int, default=200_000, help="Number of TinyStories rows to stream.")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000, help="Streaming shuffle buffer size.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = validate_model_config(load_json_config(args.config))
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = [config["unk_token"], config["bos_token"], config["eos_token"]]
    print_run_summary(
        "Tokenizer run",
        [
            ("model_name", config["name"]),
            ("dataset", "roneneldan/TinyStories"),
            ("output_dir", output_dir),
            ("max_stories", args.max_stories),
            ("vocab_size", config["vocab_size"]),
        ],
    )

    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    def text_iterator():
        count = 0
        for row in dataset:
            yield row["text"]
            count += 1
            if count >= args.max_stories:
                break

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=config["vocab_size"],
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(output_dir / "tokenizer.json"))

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(output_dir / "tokenizer.json"),
        bos_token=config["bos_token"],
        eos_token=config["eos_token"],
        unk_token=config["unk_token"],
        pad_token=config["pad_token"],
    )
    hf_tokenizer.save_pretrained(output_dir)

    metadata = {
        "dataset": "roneneldan/TinyStories",
        "max_stories": args.max_stories,
        "vocab_size": config["vocab_size"],
        "special_tokens": special_tokens,
        "seed": args.seed,
    }
    (output_dir / "tokenizer_extra.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved_tokenizer: {output_dir}")


if __name__ == "__main__":
    main()
