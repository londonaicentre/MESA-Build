"""
hf_estimator.py

Orchestrate LoRA fine-tuning on SageMaker using HuggingFace estimator
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel
from sagemaker.huggingface import HuggingFace
from sagemaker.estimator import _TrainingJob

from finetune.trainingdata_handler import TrainingDataHandler
from utils.aws import AWS
from utils.prompt import BasePromptBuilder

logger = logging.getLogger(__name__)


class HuggingFaceLoRATrainer:
    """Orchestrate LoRA fine-tuning on SageMaker using HuggingFace estimator.

    Args:
        schema: Pydantic schema class for validation
        prompt_builder: Prompt builder instance
        training_batch_names: List of S3 training batch folder names
        hyperparameters: Training hyperparameters dict (base_model, num_epochs, learning_rate, etc.)
        aws_config: AWS configuration dict with bucket, region, role
        description: Job description for naming
        instance_type: SageMaker instance type
        instance_count: Number of instances
        transformers_version: HuggingFace transformers version
        pytorch_version: PyTorch version
        py_version: Python version
    """

    def __init__(
        self,
        schema: type[BaseModel],
        prompt_builder: BasePromptBuilder,
        training_batch_names: list[str],
        hyperparameters: dict[str, Any],
        aws_config: dict[str, str],
        description: str,
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
        transformers_version: str = "4.36",
        pytorch_version: str = "2.1",
        py_version: str = "py310",
    ):
        self.schema = schema
        self.prompt_builder = prompt_builder
        self.training_batch_names = training_batch_names
        self.hyperparameters = hyperparameters
        self.aws_config = aws_config
        self.description = description
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.transformers_version = transformers_version
        self.pytorch_version = pytorch_version
        self.py_version = py_version

        # job ID
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.job_id = (
            f"{timestamp}-{description}"  # sagemaker does not like underscores!
        )

        # pass from an aws config dict
        self.bucket = aws_config["bucket"]
        self.region = aws_config["region"]
        self.role = aws_config["role"]
        self.s3_input_path = f"jobs/train/{self.job_id}/input"
        self.s3_output_path = f"s3://{self.bucket}/jobs/train/{self.job_id}/output"  # sagemaker expects full uri

    def prepare_data(self) -> str:
        """
        Prepare and upload training data to S3.

        Returns:
            S3 path to training data
        """
        logger.info(f"Preparing training data for job: {self.job_id}")

        train_jsonl = TrainingDataHandler.prepare(
            schema=self.schema,
            system_prompt=self.prompt_builder.build_main_prompt(),
            training_batch_names=self.training_batch_names,
            bucket=self.bucket,
            s3_prefix="trainingdata",
            output_file="train.jsonl",
            region=self.region,
            shuffle=True,
        )

        logger.info(f"Uploading to S3: {self.s3_input_path}")
        AWS.upload_file(
            region_name=self.region,
            file_name=train_jsonl,
            bucket=self.bucket,
            object_name="train.jsonl",
            path=self.s3_input_path,
        )

        return f"s3://{self.bucket}/{self.s3_input_path}"

    def launch_job(self, training_s3_path: str) -> str:
        """
        Launch SageMaker training job.

        Args:
            training_s3_path: S3 path to training data

        Returns:
            SageMaker job name
        """
        logger.info("Configuring SageMaker HuggingFace estimator")

        scripts_dir = Path(__file__).parent / "scripts"

        estimator = HuggingFace(
            entry_point="train_lora.py",
            source_dir=str(scripts_dir),
            role=self.role,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            transformers_version=self.transformers_version,
            pytorch_version=self.pytorch_version,
            py_version=self.py_version,
            output_path=self.s3_output_path,
            base_job_name=f"mesa-{self.job_id}",
            hyperparameters=self.hyperparameters,
        )

        logger.info("Launching SageMaker training job")
        estimator.fit({"training": training_s3_path}, wait=False)

        training_job = cast(_TrainingJob | None, estimator.latest_training_job)
        if training_job is None:
            raise RuntimeError("SageMaker training job failed to launch") 
        job_name = training_job.name
        logger.info(f"Job launched: {job_name}")

        return job_name

    def run(self) -> str:
        """
        Prepare data and launch training job.

        Returns:
            SageMaker job name
        """
        print(f"Starting training job: {self.job_id}")
        print("Preparing training data...")

        training_s3_path = self.prepare_data()

        print("Launching SageMaker job...")
        job_name = self.launch_job(training_s3_path)

        print(f"Job launched: {job_name}")

        return job_name
