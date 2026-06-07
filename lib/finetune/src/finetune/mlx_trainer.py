"""
mlx_trainer.py

Orchestrate LoRA fine-tuning locally on Apple Silicon using the mlx_lm CLIs.
"""

import logging
import subprocess
from pathlib import Path

import yaml
from mesa_types.model_card import ModelCard
from pydantic import BaseModel

from finetune._common_utils import (
    archive_and_upload,
    make_job_id,
)
from finetune.config import load_config, to_mlx_config
from finetune.trainingdata_handler import TrainingDataHandler
from utils.prompt import BasePromptBuilder

logger = logging.getLogger(__name__)


class MLXLoRATrainer:
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
    ):
        self.schema = schema
        self.prompt_builder = prompt_builder
        self.training_batch_names = training_batch_names
        self.config_path = config_path
        self.aws_config = aws_config
        self.model_name = model_name
        self.description = description
        self.work_dir = work_dir
        self.quantize = quantize

        # job ID
        self.job_id = make_job_id(description)

        # pass from an aws config dict (role unused locally)
        self.bucket = aws_config["bucket"]
        self.region = aws_config["region"]
        self.role = aws_config.get("role", "")

        # local working paths under {work_dir}/{description}/
        self.model_folder = f"{work_dir}/{description}"
        self.data_dir = f"{self.model_folder}/data"
        self.adapter_dir = f"{self.model_folder}/adapter"
        self.target_dir = f"{self.model_folder}/target"
        self.mlx_dir = f"{self.model_folder}/mlx"
        self.resolved_config_path = f"{self.model_folder}/mlx_lora_config.resolved.yaml"

        # neutral config drives training; iters derives from num_samples at _write_config time
        self.config = load_config(config_path)
        self.base_model = self.config.training.base_model
        self.num_samples: int | None = None

    def prepare_data(self) -> str:
        """Prepare training data into a local directory for mlx_lm.

        Returns:
            Path to the local data directory.
        """
        logger.info(f"Preparing training data for job: {self.job_id}")

        data_path = Path(self.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        train_jsonl = data_path / "train.jsonl"
        TrainingDataHandler.prepare(
            schema=self.schema,
            system_prompt=self.prompt_builder.build_main_prompt(),
            training_batch_names=self.training_batch_names,
            bucket=self.bucket,
            s3_prefix="trainingdata",
            output_file=str(train_jsonl),
            region=self.region,
            shuffle=True,
        )

        # mlx trains by iters (derived from sample count); count non-empty lines
        self.num_samples = sum(
            1 for line in train_jsonl.read_text().splitlines() if line.strip()
        )

        logger.info(
            f"Prepared {self.num_samples} training samples in: {self.data_dir}"
        )
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
        config = to_mlx_config(self.config, num_samples=self.num_samples)
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

    def post_process(self, model_card: ModelCard) -> None:
        """Fuse, optionally convert, and upload the fine-tuned model.

        Args:
            model_card: Model card metadata.
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
        archive_and_upload(
            target_folder=str(target_folder),
            model_card=model_card,
            model_name=self.model_name,
            region=self.region,
        )
