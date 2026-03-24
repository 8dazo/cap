#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from cap.data import iter_hackernews_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview filtered Hacker News text examples.")
    parser.add_argument("--config", required=True, help="Path to training config JSON.")
    parser.add_argument("--preview", type=int, default=10, help="Number of examples to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    dataset_cfg = cfg["dataset"]

    for index, text in enumerate(
        iter_hackernews_text(
            dataset_name=dataset_cfg["name"],
            subset=dataset_cfg.get("subset"),
            split=dataset_cfg["split"],
            seed=cfg["seed"],
            max_examples=cfg["max_train_examples"] or None,
            include_comments=dataset_cfg.get("include_comments", True),
            include_stories=dataset_cfg.get("include_stories", True),
            min_comment_chars=dataset_cfg.get("min_comment_chars", 120),
            min_story_title_chars=dataset_cfg.get("min_story_title_chars", 20),
            year_start=dataset_cfg.get("year_start"),
            year_end=dataset_cfg.get("year_end"),
        ),
        start=1,
    ):
        print(f"[sample {index}] {text}\n")
        if index >= args.preview:
            break


if __name__ == "__main__":
    main()
