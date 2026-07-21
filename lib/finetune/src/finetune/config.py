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
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base for the config models: reject unknown keys."""

    model_config = ConfigDict(extra="forbid")


class LoRAConfig(StrictModel):
    """LoRA hyperparameters meaningful to both trainers."""

    MODULE_PARENTS: ClassVar[dict[str, str]] = {
        "q_proj": "self_attn",
        "k_proj": "self_attn",
        "v_proj": "self_attn",
        "o_proj": "self_attn",
        "gate_proj": "mlp",
        "up_proj": "mlp",
        "down_proj": "mlp",
    }

    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]

    @field_validator("target_modules")
    @classmethod
    def _validate_target_modules(cls, target_modules: list[str]) -> list[str]:
        unknown = set(target_modules) - LoRAConfig.MODULE_PARENTS.keys()
        if unknown:
            raise ValueError(f"unknown LoRA target modules: {sorted(unknown)}")
        return target_modules

    def to_mlx_keys(self) -> list[str]:
        """Prefix each target module with its parent block for mlx_lm.

        Returns:
            list[str]: Fully qualified module paths, e.g. ``self_attn.q_proj``.

        """
        return [
            f"{LoRAConfig.MODULE_PARENTS[module]}.{module}"
            for module in self.target_modules
        ]


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
    grad_accumulation_steps: int = 1
    grad_checkpoint: bool = False
    weight_decay: float | None = None
    lr_scheduler_type: Literal["cosine"] | None = None
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    lr_schedule: dict[str, Any] | None = (
        None  # raw passthrough mlx_lm block; takes priority over lr_scheduler_type
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
        - keys are the target modules prefixed with their parent block
          (self_attn for q/k/v/o_proj, mlp for gate/up/down_proj)
        - lr_scheduler_type builds a cosine_decay schedule over the derived
          iters, unless a raw lr_schedule passthrough is also given
        - warmup_ratio * iters gives the schedule's warmup step count

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
            "grad_accumulation_steps": training.mlx.grad_accumulation_steps,
            "grad_checkpoint": training.mlx.grad_checkpoint,
            "lora_parameters": {
                "keys": training.lora.to_mlx_keys(),
                "rank": training.lora.rank,
                "scale": training.lora.alpha / training.lora.rank,
                "dropout": training.lora.dropout,
            },
        }
        if training.mlx.weight_decay is not None:
            out["optimizer_config"] = {"weight_decay": training.mlx.weight_decay}
        if training.mlx.lr_schedule is not None:
            out["lr_schedule"] = training.mlx.lr_schedule
        elif training.mlx.lr_scheduler_type is not None:
            out["lr_schedule"] = {
                "name": "cosine_decay",
                "arguments": [training.learning_rate, iters],
                "warmup": round(training.mlx.warmup_ratio * iters),
            }
        return out
