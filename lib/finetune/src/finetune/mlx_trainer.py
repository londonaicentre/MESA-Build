"""
mlx_trainer.py

Orchestrate LoRA fine-tuning locally on Apple Silicon using the mlx_lm CLIs.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml
from mesa_types.model_card import ModelCard
from pydantic import BaseModel

from finetune.config import FinetuneConfig
from finetune.trainer import LoRATrainer
from utils.prompt import BasePromptBuilder

logger = logging.getLogger(__name__)


class MLXLoRATrainer(LoRATrainer):
    """Orchestrate LoRA fine-tuning locally on Apple Silicon using mlx_lm.

    Args:
        schema: Pydantic schema class for validation.
        prompt_builder: Prompt builder instance.
        training_batch_names: List of S3 training batch folder names.
        config_path: Path to the neutral ``config.yaml`` holding training parameters.
        aws_config: AWS configuration dict with ``bucket``, ``region``, ``role``
            (``role`` is unused for local training).
        model_name: Model name used for the model card and the uploaded archive.
        description: Job description, used for naming and the working directory.
        work_dir: Local working/output root. Defaults to ``data/models``.
        quantize: Optional MLX quantisation for ``convert`` (``None`` | ``"q4"`` | ``"q8"``).
        config: Pre-loaded config, bypassing ``config_path``. Used when rebuilding a
            trainer from a serialised spec; defaults to loading ``config_path``.
    """

    def __init__(
        self,
        schema: type[BaseModel],
        prompt_builder: BasePromptBuilder,
        training_batch_names: list[str],
        config_path: str,
        aws_config: dict[str, str],
        model_name: str,
        description: str,
        work_dir: str = "data/models",
        quantize: str | None = None,
        config: FinetuneConfig | None = None,
    ):
        super().__init__(
            schema,
            prompt_builder,
            training_batch_names,
            config_path,
            aws_config,
            model_name,
            description,
            config,
        )
        self.work_dir = work_dir
        self.quantize = quantize

        # local working paths under {work_dir}/{description}/
        self.model_folder = f"{work_dir}/{description}"
        self.data_dir = f"{self.model_folder}/data"
        self.adapter_dir = f"{self.model_folder}/adapter"
        self.target_dir = f"{self.model_folder}/target"
        self.mlx_dir = f"{self.model_folder}/mlx"
        self.resolved_config_path = f"{self.model_folder}/mlx_lora_config.resolved.yaml"

        # neutral config (loaded by the base) drives training; iters derives from
        # num_samples at _write_config time
        self.num_samples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the trainer's state, extending the base with MLX fields.

        Returns:
            dict[str, Any]: The trainer state, ready for ``json.dumps``.

        """
        return {
            **super().to_dict(),
            "work_dir": self.work_dir,
            "quantize": self.quantize,
            "num_samples": self.num_samples,
        }

    @classmethod
    def _constructor_kwargs(cls, data: dict[str, Any]) -> dict[str, Any]:
        return {
            **super()._constructor_kwargs(data),
            "work_dir": data["work_dir"],
            "quantize": data["quantize"],
        }

    def _restore_runtime(self, data: dict[str, Any]) -> None:
        super()._restore_runtime(data)
        self.num_samples = data["num_samples"]

    def prepare_data(self) -> str:
        """Prepare training data into a local directory for mlx_lm.

        Returns:
            Path to the local data directory.
        """
        logger.info(f"Preparing training data for job: {self.job_id}")

        data_path = Path(self.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        train_jsonl = data_path / "train.jsonl"
        self._prepare_training_data(str(train_jsonl))

        # mlx trains by iters (derived from sample count); count non-empty lines
        self.num_samples = sum(
            1 for line in train_jsonl.read_text().splitlines() if line.strip()
        )

        logger.info(f"Prepared {self.num_samples} training samples in: {self.data_dir}")
        return self.data_dir

    def _write_config(self, data_dir: str) -> str:
        """Write a resolved mlx_lm config for this run.

        Args:
            data_dir: The prepared training-data directory.

        Returns:
            Path to the resolved config file.
        """
        if self.num_samples is None:
            raise ValueError(
                "prepare_data must run before _write_config (num_samples unset)"
            )
        config = self.config.to_mlx_config(self.num_samples)
        config["data"] = data_dir
        config["adapter_path"] = self.adapter_dir

        resolved_path = Path(self.resolved_config_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Wrote resolved config to: {self.resolved_config_path}")
        return self.resolved_config_path

    def train(self, config_path: str) -> None:
        """Run mlx_lm LoRA training.

        Args:
            config_path: Path to the resolved mlx_lm YAML config.
        """
        logger.info("Launching mlx_lm.lora training")
        subprocess.run(["mlx_lm.lora", "--config", config_path], check=True)
        logger.info("mlx_lm.lora training complete")

    def run(self) -> str:
        """Prepare data, write the resolved config, and train.

        Returns:
            The job ID.
        """
        print(f"Starting training job: {self.job_id}")
        print("Preparing training data...")
        data_dir = self.prepare_data()

        print("Writing resolved mlx config...")
        config_path = self._write_config(data_dir)

        print("Launching mlx_lm.lora training...")
        self.train(config_path)

        print(f"Job complete: {self.job_id}")
        return self.job_id

    def fuse(self, target_folder: str) -> bool:
        """Fuse (merge) the LoRA adapter into the base model.

        Args:
            target_folder: Folder to save the fused model to.

        Returns:
            True if fuse successful.
        """
        if Path(f"{target_folder}/model.safetensors").exists():
            return True
        logger.info("Fusing adapter with base model via mlx_lm.fuse")
        subprocess.run(
            [
                "mlx_lm.fuse",
                "--model",
                self.base_model,
                "--adapter-path",
                self.adapter_dir,
                "--save-path",
                target_folder,
            ],
            check=True,
        )
        return True

    def convert(self, target_folder: str, mlx_folder: str) -> bool:
        """Optionally convert the fused model to MLX format.

        Args:
            target_folder: Folder containing the fused (HF-format) model.
            mlx_folder: Folder to save the MLX-format model to.

        Returns:
            True if conversion successful.
        """
        logger.info("Converting fused model to MLX format via mlx_lm.convert")
        cmd = [
            "mlx_lm.convert",
            "--hf-path",
            target_folder,
            "--mlx-path",
            mlx_folder,
        ]
        if self.quantize:
            cmd.extend(["-q", self.quantize])
        subprocess.run(cmd, check=True)
        return True

    def post_process(self, model_card: ModelCard, push_public: bool = False) -> None:
        """Fuse, optionally convert, and upload the fine-tuned model.

        The primary publish target is the build bucket (unpacked, under
        models/{model_name}/{model_name}_{v}/)

        Set push_public=True to also push tarball to the public bucket.

        Args:
            model_card: Model card metadata.
            push_public: Also upload the public tarball. Defaults to False.
        """
        target_folder = Path(self.target_dir)
        target_folder.mkdir(parents=True, exist_ok=True)
        if not self.fuse(str(target_folder)):
            raise ValueError("fusing with base model failed")
        if self.quantize is not None:
            mlx_folder = Path(self.mlx_dir)
            mlx_folder.mkdir(parents=True, exist_ok=True)
            if not self.convert(str(target_folder), str(mlx_folder)):
                raise ValueError("converting to MLX format failed")
        self._publish(str(target_folder), model_card, push_public)
