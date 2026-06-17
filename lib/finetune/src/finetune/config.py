"""
config.py

Defines standard firmnat for training parameters, loaded/validated
with pydantic, then translated into each trainer's native form
including MLX and HF estimators.

An example config file can be found in /config/config.yaml

Operational args stay per-trainer constructor inputs:
E.g. instance type, AWS config, work_dir, quantization etc
"""

import math
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for the config models: reject unknown keys."""

    model_config = ConfigDict(extra="forbid")


class LoRAConfig(StrictModel):
    """LoRA hyperparameters meaningful to both trainers."""

    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]


class MLXOverrides(StrictModel):
    """
    MLX-only params, consumed ONLY by to_mlx_config()
    """

    iters: int | None = None  # if set, OVERRIDES the epochs-derived value
    num_layers: int = -1
    seed: int = 0
    save_every: int = 100
    steps_per_report: int = 10
    steps_per_eval: int = 200
    val_batches: int = 25
    lr_schedule: dict[str, Any] | None = (
        None  # passthrough mlx_lm block; omitted if None
    )


class TrainingConfig(StrictModel):
    """
    Base training param class
    """

    base_model: str
    epochs: int
    learning_rate: float
    batch_size: int
    max_seq_length: int
    lora: LoRAConfig
    mlx: MLXOverrides = MLXOverrides()


class FinetuneConfig(StrictModel):
    """
    Top-level neutral config
    """

    training: TrainingConfig

    @classmethod
    def load(cls, path: str | Path) -> "FinetuneConfig":
        """Load and validate a neutral config from a YAML file.

        Args:
            path (str | Path): Path to the YAML config file.

        Returns:
            FinetuneConfig: The validated config.

        Raises:
            pydantic.ValidationError: On missing required keys or unknown keys.

        """
        with open(path) as config_file:
            return cls(**yaml.safe_load(config_file))

    @classmethod
    def load_default(cls) -> "FinetuneConfig":
        """Load the neutral config shipped with the package.

        Returns:
            FinetuneConfig: The validated default config.

        """
        return cls(
            **yaml.safe_load(
                files("finetune").joinpath("config/config.yaml").read_text()
            )
        )

    def to_hf_hyperparameters(self) -> dict[str, Any]:
        """Translate the neutral config into a dict for the HF estimator.

        Returns:
            dict[str, Any]: Hyperparameters keyed for the SageMaker entry point.

        """
        training = self.training
        return {
            "base_model": training.base_model,
            "num_epochs": training.epochs,
            "learning_rate": training.learning_rate,
            "lora_r": training.lora.rank,
            "lora_alpha": training.lora.alpha,
            "lora_dropout": training.lora.dropout,
            "lora_target_modules": ",".join(
                training.lora.target_modules
            ),  ## format for --lora_target_modules
            "per_device_train_batch_size": training.batch_size,
            "max_seq_length": training.max_seq_length,
        }

    def to_mlx_config(self, num_samples: int) -> dict[str, Any]:
        """Translate the neutral config into an mlx_lm-native config dict.

        Returns everything EXCEPT the runtime-injected data / adapter_path,
        which stay the trainer's responsibility.

        Specific args:
        - iters are derived from samples / batch size if not specified
        - scale is alpha / rank
        - keys are the target modules prefixed with self_attn

        Args:
            num_samples (int): Number of training samples, used to derive iters.

        Returns:
            dict[str, Any]: Config consumable by ``mlx_lm.lora``.

        """
        training = self.training
        iters = (
            training.mlx.iters
            if training.mlx.iters is not None
            else math.ceil(num_samples / training.batch_size) * training.epochs
        )
        out: dict[str, Any] = {
            "model": training.base_model,
            "train": True,  # MLX-runner literal constants (not in the neutral config)
            "fine_tune_type": "lora",
            "optimizer": "adamw",
            "seed": training.mlx.seed,
            "num_layers": training.mlx.num_layers,
            "batch_size": training.batch_size,
            "iters": iters,
            "learning_rate": training.learning_rate,
            "max_seq_length": training.max_seq_length,
            "save_every": training.mlx.save_every,
            "steps_per_report": training.mlx.steps_per_report,
            "steps_per_eval": training.mlx.steps_per_eval,
            "val_batches": training.mlx.val_batches,
            "lora_parameters": {
                "keys": [
                    f"self_attn.{module}" for module in training.lora.target_modules
                ],
                "rank": training.lora.rank,
                "scale": training.lora.alpha / training.lora.rank,
                "dropout": training.lora.dropout,
            },
        }
        if training.mlx.lr_schedule is not None:
            out["lr_schedule"] = training.mlx.lr_schedule
        return out
