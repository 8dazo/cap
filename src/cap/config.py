from __future__ import annotations

import json
from pathlib import Path


class ConfigError(ValueError):
    """Raised when a tutorial config file is missing required values or is malformed."""


def load_json_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc


def _require_keys(config: dict, required_keys: list[str], label: str) -> None:
    missing = [key for key in required_keys if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"{label} is missing required keys: {joined}")


def validate_model_config(config: dict) -> dict:
    _require_keys(
        config,
        [
            "name",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "hidden_act",
            "max_position_embeddings",
            "rope_theta",
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
        ],
        "Model config",
    )
    for key in ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "max_position_embeddings"]:
        if int(config[key]) <= 0:
            raise ConfigError(f"Model config value must be > 0: {key}={config[key]}")
    return config


def validate_train_config(config: dict) -> dict:
    _require_keys(
        config,
        [
            "output_dir",
            "tokenizer_dir",
            "dataset",
            "sequence_length",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "total_tokens",
            "warmup_steps",
            "learning_rate",
            "min_learning_rate",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "grad_clip",
            "eval_every_steps",
            "eval_batches",
            "save_every_steps",
            "max_train_examples",
            "seed",
            "use_gradient_checkpointing",
        ],
        "Train config",
    )
    dataset = config["dataset"]
    _require_keys(dataset, ["name", "split", "streaming"], "Train config dataset")
    for key in ["sequence_length", "micro_batch_size", "gradient_accumulation_steps", "total_tokens", "eval_every_steps", "eval_batches", "save_every_steps"]:
        if int(config[key]) <= 0:
            raise ConfigError(f"Train config value must be > 0: {key}={config[key]}")
    if float(config["learning_rate"]) <= 0:
        raise ConfigError("Train config learning_rate must be > 0")
    if float(config["min_learning_rate"]) < 0:
        raise ConfigError("Train config min_learning_rate must be >= 0")
    if dataset["name"] not in {"roneneldan/TinyStories", "open-index/hacker-news"}:
        raise ConfigError(f"Unsupported dataset name: {dataset['name']}")
    return config


def print_run_summary(title: str, rows: list[tuple[str, object]]) -> None:
    print(f"{title}:")
    for key, value in rows:
        print(f"  {key}: {value}")
