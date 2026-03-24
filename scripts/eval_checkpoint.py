#!/usr/bin/env python3
import argparse
import math

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cap.data import PackedTinyStoriesDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Cap checkpoint on a bounded TinyStories stream.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory to load.")
    parser.add_argument("--sequence-length", type=int, default=256, help="Packed evaluation sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--eval-batches", type=int, default=20, help="Number of evaluation batches to average.")
    parser.add_argument("--seed", type=int, default=123, help="Streaming seed for evaluation sampling.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--max-examples", type=int, default=5000, help="Maximum streamed rows to consume.")
    return parser.parse_args()


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)
    model.eval()

    dataset = PackedTinyStoriesDataset(
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        split=args.split,
        seed=args.seed,
        max_examples=args.max_examples,
        dataset_name="roneneldan/TinyStories",
        subset=None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    losses = []
    print("Evaluation run:")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  device: {device}")
    print(f"  sequence_length: {args.sequence_length}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  eval_batches: {args.eval_batches}")

    for batch_index, (x, y) in enumerate(loader, start=1):
        if batch_index > args.eval_batches:
            break
        x = x.to(device)
        y = y.to(device)
        out = model(input_ids=x, labels=y)
        loss = float(out.loss.detach().float().item())
        losses.append(loss)
        print(f"eval_batch={batch_index}/{args.eval_batches} loss={loss:.4f}", flush=True)

    if not losses:
        raise SystemExit("No evaluation batches were produced.")

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    print(f"eval_loss={mean_loss:.4f}")
    print(f"eval_ppl={perplexity:.2f}")


if __name__ == "__main__":
    main()
