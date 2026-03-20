"""
hf_estimator.py

Orchestrate LoRA fine-tuning on SageMaker using HuggingFace estimator
"""

import logging
from datetime import datetime
from pathlib import Path
import tarfile
from typing import Any, cast

from mesa_types.model_card import ModelCard
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
        model_name: str,
        description: str,
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
        transformers_version: str = "4.56",
        pytorch_version: str = "2.8",
        py_version: str = "py312",
    ):
        self.schema = schema
        self.prompt_builder = prompt_builder
        self.training_batch_names = training_batch_names
        self.hyperparameters = hyperparameters
        self.aws_config = aws_config
        self.model_name = model_name
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
        self.s3_output_path = f"jobs/train/{self.job_id}/output"
        self.s3_full_output_path = (
            f"s3://{self.bucket}/{self.s3_output_path}"  # sagemaker expects full uri
        )
        self.last_job_name: str | None = None

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
            code_location=f"s3://{self.bucket}/jobs/train/{self.job_id}",
            role=self.role,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            transformers_version=self.transformers_version,
            pytorch_version=self.pytorch_version,
            py_version=self.py_version,
            output_path=self.s3_full_output_path,
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
        self.last_job_name = job_name
        return job_name

    def download_output(
        self, source_folder: str, s3_output_path: str, job_name: str
    ) -> bool:
        """Download and extract training output from S3.

        Args:
            source_folder (str): Local folder to download model archive to.
            s3_output_path (str): S3 path prefix for output.
            job_name (str): SageMaker job name.

        Returns:
            bool: True if download and extraction successful.

        """
        source_file: Path = Path(f"{source_folder}/model.tar.gz")
        if source_file.exists():
            return True
        success = AWS.download_file(
            region_name=self.region,
            bucket=self.bucket,
            file_name=str(source_file),
            object_name="model.tar.gz",
            path=f"{s3_output_path}/{job_name}/output",
        )
        if not success:
            raise ValueError("Failed to download training output")
        with tarfile.open(source_file, "r:*") as tar:
            tar.extractall(source_file.parent)
        return True

    def merge(self, source_folder: str, target_folder: str) -> bool:
        """Merge LoRA weights with base model and save full model.

        Args:
            source_folder (str): Folder containing LoRA adapter weights.
            target_folder (str): Folder to save merged model to.

        Returns:
            bool: True if merge successful.

        """
        if Path(f"{target_folder}/model.safetensors").exists():
            return True
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            self.hyperparameters["base_model"], dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(
            base, source_folder, autocast_adapter_dtype=False
        )
        merged = model.merge_and_unload()
        merged.save_pretrained(target_folder)
        AutoTokenizer.from_pretrained(source_folder).save_pretrained(target_folder)
        return True

    def upload_output(
        self,
        target_folder: str,
        model_card: ModelCard,
        bucket: str = "aicentre-nlpteam-mesa-public",
    ) -> bool:
        """Archive merged model with metadata and upload to S3.

        Args:
            target_folder (str): Folder containing merged model.
            model_card (ModelCard): Model card metadata.
            bucket (str): S3 bucket name. Defaults to 'aicentre-nlpteam-mesa-public'.

        Returns:
            bool: True if upload successful.

        """
        target_path = Path(target_folder)
        archive_name = f"{model_card.model_name}_{model_card.major}_{model_card.minor}_{model_card.patch}.tar.gz"
        archive_path = target_path.parent / archive_name
        if not archive_path.exists():
            import io

            with tarfile.open(archive_path, "w:gz") as tar:
                for item in target_path.iterdir():
                    tar.add(item, arcname=item.name)
                yaml_bytes: bytes = model_card.to_yaml_bytes()
                tarinfo: tarfile.TarInfo = tarfile.TarInfo(name="model_card.yml")
                tarinfo.size = len(yaml_bytes)
                tar.addfile(tarinfo, io.BytesIO(yaml_bytes))
                tar.add(Path(__file__).parents[2] / "LICENSE.md", arcname="LICENSE.md")
        success = AWS.upload_file(
            region_name=self.region,
            file_name=str(archive_path),
            bucket=bucket,
            object_name=archive_name,
            path=self.model_name,
        )
        if not success:
            raise ValueError("Failed to upload merged model weights")
        return True

    def create_model_card(
        self, major: int, minor: int, patch: int, model_description: str | None = None
    ) -> ModelCard:
        """Create model card with training metadata.

        Args:
            major (int): Major version number.
            minor (int): Minor version number.
            patch (int): Patch version number.
            model_description (str | None): Model description. Defaults to None (uses self.description).

        Returns:
            ModelCard: Model card instance.

        """
        return ModelCard(
            base_model_hf=self.hyperparameters["base_model"],
            model_name=self.model_name,
            major=major,
            minor=minor,
            patch=patch,
            model_description=model_description or self.description,
            training_data=[self.s3_input_path],
            output_schema=self.schema,
        )

    def post_process(
        self, model_card: ModelCard, s3_output_path: str | None, job_name: str | None
    ) -> None:
        """Download, merge and upload fine-tuned model.

        Args:
            model_card (ModelCard): Model card metadata.
            s3_output_path (str | None): S3 output path. If None, uses self.s3_output_path.
            job_name (str | None): SageMaker job name. If None, uses self.last_job_name.

        """
        model_folder = f"data/models/{self.description}"
        source_folder = Path(f"{model_folder}/source")
        source_folder.mkdir(parents=True, exist_ok=True)
        s3_output_path = s3_output_path or self.s3_output_path
        job_name = job_name or self.last_job_name
        if not job_name:
            raise ValueError("no last job available and no job name specified")
        if not self.download_output(str(source_folder), str(s3_output_path), job_name):
            raise ValueError("downloading low-rank weights failed")
        target_folder = Path(f"{model_folder}/target")
        target_folder.mkdir(parents=True, exist_ok=True)
        if not self.merge(str(source_folder), str(target_folder)):
            raise ValueError("merging with base model failed")
        self.upload_output(str(target_folder), model_card)
