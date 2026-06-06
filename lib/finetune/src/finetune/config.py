"""
config.py

Defines neutral fine-tuning config and translators.

An example config file can be found in ../config/config.yaml

Single source of truth for training parameters, shared by any trainer
including HF and local MLX. The neutral config is loaded/validated
with pydantic, then translated into each trainer's native form:
  * ``to_hf_hyperparameters`` -> the flat ``hyperparameters`` dict the SageMaker
    estimator passes to ``scripts/train_lora.py`` as CLI args.
  * ``to_mlx_config`` -> the ``mlx_lm``-native config dict consumed by
    ``mlx_lm.lora --config`` (minus the runtime-injected ``data`` / ``adapter_path``).

Operational args (instance type, AWS config, work_dir, quantize, ...) are deliberately
NOT modelled, they stay per-trainer constructor inputs
"""

import math
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class LoRAConfig(BaseModel):
    """LoRA hyperparameters meaningful to both trainers."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]


class MLXOverrides(BaseModel):
    """
    MLX-only params, consumed ONLY by ``to_mlx_config``
    """

    model_config = ConfigDict(extra="forbid")

    iters: int | None = None  # if set, OVERRIDES the epochs-derived value
    num_layers: int = -1
    seed: int = 0
    save_every: int = 100
    steps_per_report: int = 10
    steps_per_eval: int = 200
    val_batches: int = 25
    lr_schedule: dict[str, Any] | None = None  # passthrough mlx_lm block; omitted if None


class TrainingConfig(BaseModel):
    """
    Base training param class
    """

    model_config = ConfigDict(extra="forbid")

    base_model: str
    epochs: int
    learning_rate: float
    batch_size: int
    max_seq_length: int
    lora: LoRAConfig
    mlx: MLXOverrides = MLXOverrides()


class FinetuneConfig(BaseModel):
    """
    Top-level neutral config
    """

    model_config = ConfigDict(extra="forbid")

    training: TrainingConfig


def load_config(path: str | Path) -> FinetuneConfig:
    """
    Load and validate a neutral config from a YAML file.

    Raises:
        pydantic.ValidationError: on missing required keys or unknown keys
    """
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return FinetuneConfig(**data)


def load_default_config() -> FinetuneConfig:
    """
    Load the neutral config
    """
    config_text = files("finetune").joinpath("config/config.yaml").read_text()
    data: dict[str, Any] = yaml.safe_load(config_text)
    return FinetuneConfig(**data)


def to_hf_hyperparameters(cfg: FinetuneConfig) -> dict[str, Any]:
    """
    Translate the neutral config into the HF estimator ``hyperparameters`` dict
    These become CLI args for ``scripts/train_lora.py``
    """
    t = cfg.training
    return {
        "base_model": t.base_model,
        "num_epochs": t.epochs,
        "learning_rate": t.learning_rate,
        "lora_r": t.lora.rank,
        "lora_alpha": t.lora.alpha,
        "lora_dropout": t.lora.dropout,
        "lora_target_modules": ",".join(t.lora.target_modules), ## expected by ``--lora_target_modules``
        "per_device_train_batch_size": t.batch_size,
        "max_seq_length": t.max_seq_length,
    }


def to_mlx_config(cfg: FinetuneConfig, num_samples: int) -> dict[str, Any]:
    """Translate the neutral config into an ``mlx_lm``-native config dict.

    Returns everything ``mlx_lm.lora --config`` needs EXCEPT the runtime-injected
    ``data`` / ``adapter_path``, which stay the trainer's responsibility.

    ``iters`` is taken from ``mlx.iters`` if set, otherwise derived from the dataset size
    as ``ceil(num_samples / batch_size) * epochs``.

    ``scale`` is ``alpha / rank`` and
    ``keys`` are the target modules prefixed with ``self_attn.``.
    """
    t = cfg.training
    iters = (
        t.mlx.iters
        if t.mlx.iters is not None
        else math.ceil(num_samples / t.batch_size) * t.epochs
    )
    out: dict[str, Any] = {
        "model": t.base_model,
        "train": True,  # MLX-runner literal constants (not in the neutral config)
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "seed": t.mlx.seed,
        "num_layers": t.mlx.num_layers,
        "batch_size": t.batch_size,
        "iters": iters,
        "learning_rate": t.learning_rate,
        "max_seq_length": t.max_seq_length,
        "save_every": t.mlx.save_every,
        "steps_per_report": t.mlx.steps_per_report,
        "steps_per_eval": t.mlx.steps_per_eval,
        "val_batches": t.mlx.val_batches,
        "lora_parameters": {
            "keys": [f"self_attn.{m}" for m in t.lora.target_modules],
            "rank": t.lora.rank,
            "scale": t.lora.alpha / t.lora.rank,
            "dropout": t.lora.dropout,
        },
    }
    if t.mlx.lr_schedule is not None:
        out["lr_schedule"] = t.mlx.lr_schedule
    return out
