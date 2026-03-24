from __future__ import annotations

from datetime import datetime
from typing import Iterable, Iterator, Optional

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset


def _year_ok(timestamp, year_start: Optional[int], year_end: Optional[int]) -> bool:
    if timestamp is None:
        return False
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    else:
        dt = timestamp
    if year_start is not None and dt.year < year_start:
        return False
    if year_end is not None and dt.year > year_end:
        return False
    return True


def iter_tinystories_text(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    seed: int,
    max_examples: Optional[int] = None,
) -> Iterator[str]:
    dataset = load_dataset(dataset_name, name=subset, split=split, streaming=True)
    dataset = dataset.shuffle(buffer_size=10_000, seed=seed)
    count = 0
    for row in dataset:
        yield row["text"]
        count += 1
        if max_examples and count >= max_examples:
            break


def iter_hackernews_text(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    seed: int,
    max_examples: Optional[int],
    include_comments: bool,
    include_stories: bool,
    min_comment_chars: int,
    min_story_title_chars: int,
    year_start: Optional[int],
    year_end: Optional[int],
) -> Iterator[str]:
    dataset = load_dataset(dataset_name, name=subset, split=split, streaming=True)
    dataset = dataset.shuffle(buffer_size=50_000, seed=seed)
    count = 0
    for row in dataset:
        if row.get("deleted") or row.get("dead"):
            continue
        if not _year_ok(row.get("time"), year_start, year_end):
            continue

        row_type = row.get("type")
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()

        if include_stories and row_type == 1 and len(title) >= min_story_title_chars:
            url = (row.get("url") or "").strip()
            story_text = title if not url else f"{title}\n{url}"
            yield story_text
            count += 1
        elif include_comments and row_type == 2 and len(text) >= min_comment_chars:
            yield text
            count += 1

        if max_examples and count >= max_examples:
            break


class _PackedTextDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        sequence_length: int,
        seed: int,
        split: str,
        max_examples: Optional[int],
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.seed = seed
        self.split = split
        self.max_examples = max_examples

    def text_iter(self) -> Iterable[str]:
        raise NotImplementedError

    def __iter__(self):
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        buffer = []
        for text in self.text_iter():
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend([bos] + ids + [eos])
            while len(buffer) >= self.sequence_length + 1:
                block = buffer[: self.sequence_length + 1]
                buffer = buffer[self.sequence_length + 1 :]
                x = torch.tensor(block[:-1], dtype=torch.long)
                y = torch.tensor(block[1:], dtype=torch.long)
                yield x, y


class PackedTinyStoriesDataset(_PackedTextDataset):
    def __init__(self, dataset_name: str, subset: Optional[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.subset = subset

    def text_iter(self) -> Iterable[str]:
        return iter_tinystories_text(
            dataset_name=self.dataset_name,
            subset=self.subset,
            split=self.split,
            seed=self.seed,
            max_examples=self.max_examples,
        )


class PackedHackerNewsDataset(_PackedTextDataset):
    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str],
        include_comments: bool,
        include_stories: bool,
        min_comment_chars: int,
        min_story_title_chars: int,
        year_start: Optional[int],
        year_end: Optional[int],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.subset = subset
        self.include_comments = include_comments
        self.include_stories = include_stories
        self.min_comment_chars = min_comment_chars
        self.min_story_title_chars = min_story_title_chars
        self.year_start = year_start
        self.year_end = year_end

    def text_iter(self) -> Iterable[str]:
        return iter_hackernews_text(
            dataset_name=self.dataset_name,
            subset=self.subset,
            split=self.split,
            seed=self.seed,
            max_examples=self.max_examples,
            include_comments=self.include_comments,
            include_stories=self.include_stories,
            min_comment_chars=self.min_comment_chars,
            min_story_title_chars=self.min_story_title_chars,
            year_start=self.year_start,
            year_end=self.year_end,
        )
