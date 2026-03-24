#!/usr/bin/env python3
import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cap.config import ConfigError, load_json_config, print_run_summary, validate_model_config, validate_train_config
from cap.data import PackedHackerNewsDataset, PackedTinyStoriesDataset
from cap.modeling import build_llama_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Cap tutorial model.")
    parser.add_argument("--model-config", required=True, help="Path to model config JSON.")
    parser.add_argument("--train-config", required=True, help="Path to training config JSON.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional base checkpoint directory for continued pretraining.",
    )
    return parser.parse_args()


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_loader(tokenizer, train_config: dict, split: str):
    dataset_cfg = train_config["dataset"]
    common_kwargs = {
        "tokenizer": tokenizer,
        "sequence_length": train_config["sequence_length"],
        "split": split,
        "seed": train_config["seed"],
        "max_examples": train_config["max_train_examples"] or None,
    }
    if dataset_cfg["name"] == "roneneldan/TinyStories":
        dataset = PackedTinyStoriesDataset(
            dataset_name=dataset_cfg["name"],
            subset=dataset_cfg.get("subset"),
            **common_kwargs,
        )
    elif dataset_cfg["name"] == "open-index/hacker-news":
        dataset = PackedHackerNewsDataset(
            dataset_name=dataset_cfg["name"],
            subset=dataset_cfg.get("subset"),
            include_comments=dataset_cfg.get("include_comments", True),
            include_stories=dataset_cfg.get("include_stories", True),
            min_comment_chars=dataset_cfg.get("min_comment_chars", 120),
            min_story_title_chars=dataset_cfg.get("min_story_title_chars", 20),
            year_start=dataset_cfg.get("year_start"),
            year_end=dataset_cfg.get("year_end"),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_cfg['name']}")
    return DataLoader(dataset, batch_size=train_config["micro_batch_size"], num_workers=0)


def lr_schedule(step: int, total_steps: int, cfg: dict) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["learning_rate"] * step / max(1, cfg["warmup_steps"])
    progress = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg["min_learning_rate"] + cosine * (cfg["learning_rate"] - cfg["min_learning_rate"])


@torch.no_grad()
def evaluate(model, loader, device: str, amp_context, max_batches: int):
    model.eval()
    losses = []
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with amp_context:
            out = model(input_ids=x, labels=y)
        losses.append(float(out.loss.detach().float().item()))
    model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    return mean_loss, math.exp(mean_loss)


def main() -> None:
    args = parse_args()
    try:
        model_config = validate_model_config(load_json_config(args.model_config))
        train_config = validate_train_config(load_json_config(args.train_config))
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(train_config["tokenizer_dir"])
    device = pick_device()
    print_run_summary(
        "Training run",
        [
            ("model_name", model_config["name"]),
            ("dataset", train_config["dataset"]["name"]),
            ("split", train_config["dataset"]["split"]),
            ("output_dir", output_dir),
            ("tokenizer_dir", train_config["tokenizer_dir"]),
            ("device", device),
            ("sequence_length", train_config["sequence_length"]),
            ("micro_batch_size", train_config["micro_batch_size"]),
            ("gradient_accumulation_steps", train_config["gradient_accumulation_steps"]),
            ("total_tokens", train_config["total_tokens"]),
            ("init_checkpoint", args.init_checkpoint or "<none>"),
        ],
    )

    if device in {"mps", "cuda"}:
        amp_context = torch.autocast(device_type=device, dtype=torch.float16)
    else:
        amp_context = nullcontext()

    if args.init_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(args.init_checkpoint).to(device)
    else:
        model = build_llama_model(model_config, tokenizer).to(device)
    if train_config.get("use_gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_config["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train_config["learning_rate"],
        betas=(train_config["adam_beta1"], train_config["adam_beta2"]),
        eps=train_config["adam_epsilon"],
    )

    tokens_per_step = (
        train_config["sequence_length"]
        * train_config["micro_batch_size"]
        * train_config["gradient_accumulation_steps"]
    )
    total_steps = math.ceil(train_config["total_tokens"] / tokens_per_step)
    print_run_summary(
        "Training schedule",
        [
            ("tokens_per_step", tokens_per_step),
            ("total_steps", total_steps),
            ("learning_rate", train_config["learning_rate"]),
            ("min_learning_rate", train_config["min_learning_rate"]),
            ("save_every_steps", train_config["save_every_steps"]),
        ],
    )

    train_loader = build_loader(tokenizer, train_config, split=train_config["dataset"]["split"])
    eval_loader = build_loader(tokenizer, train_config, split=train_config["dataset"]["split"])

    optimizer.zero_grad(set_to_none=True)
    running_tokens = 0
    started_at = time.time()

    model.train()
    for step, (x, y) in enumerate(train_loader, start=1):
        if step > total_steps:
            break

        lr = lr_schedule(step, total_steps, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x = x.to(device)
        y = y.to(device)

        with amp_context:
            out = model(input_ids=x, labels=y)
            loss = out.loss / train_config["gradient_accumulation_steps"]

        loss.backward()

        if step % train_config["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_tokens += x.numel()

        if step % 50 == 0:
            elapsed = time.time() - started_at
            tok_per_sec = running_tokens / max(elapsed, 1e-9)
            print(
                f"step={step}/{total_steps} lr={lr:.2e} "
                f"loss={loss.item() * train_config['gradient_accumulation_steps']:.4f} "
                f"tok_per_sec={tok_per_sec:.0f}"
            )

        if step % train_config["eval_every_steps"] == 0:
            eval_loss, eval_ppl = evaluate(model, eval_loader, device, amp_context, train_config["eval_batches"])
            print(f"eval_step={step} eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}")

        if step % train_config["save_every_steps"] == 0:
            checkpoint_dir = output_dir / f"checkpoint-step-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
            torch.save(
                {"step": step, "optimizer": optimizer.state_dict()},
                checkpoint_dir / "trainer_state.pt",
            )
            print(f"saved_checkpoint: {checkpoint_dir}")

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"saved_final_model: {output_dir}")


if __name__ == "__main__":
    main()
