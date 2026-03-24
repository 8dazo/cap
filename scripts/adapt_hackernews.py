#!/usr/bin/env python3
import argparse
import subprocess
import sys

from cap.config import ConfigError, load_json_config, validate_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue training from a base checkpoint on Hacker News.")
    parser.add_argument("--model-config", required=True, help="Path to model config JSON.")
    parser.add_argument("--train-config", required=True, help="Path to Hacker News train config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Base checkpoint directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        train_config = validate_train_config(load_json_config(args.train_config))
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    if train_config["dataset"]["name"] != "open-index/hacker-news":
        raise SystemExit("Hacker News adaptation requires configs/train/hackernews_adapt.json or another open-index/hacker-news config.")
    command = [
        sys.executable,
        "scripts/train_pretrain.py",
        "--model-config",
        args.model_config,
        "--train-config",
        args.train_config,
        "--init-checkpoint",
        args.checkpoint,
    ]
    print("running:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
