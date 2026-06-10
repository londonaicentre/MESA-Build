"""
hf_estimator.py

Orchestrate LoRA fine-tuning on SageMaker using HuggingFace estimator
"""

import logging
from pathlib import Path
import tarfile
from typing import cast

from mesa_types.model_card import ModelCard
from pydantic import BaseModel
from sagemaker.huggingface import HuggingFace
from sagemaker.estimator import _TrainingJob

from finetune.trainer import LoRATrainer
from utils.aws import AWS
from utils.prompt import BasePromptBuilder

logger = logging.getLogger(__name__)


class HuggingFaceLoRATrainer(LoRATrainer):
    """Orchestrate LoRA fine-tuning on SageMaker using HuggingFace estimator.

    Args:
        schema: Pydantic schema class for validation
        prompt_builder: Prompt builder instance
        training_batch_names: List of S3 training batch folder names
        config_path: Path to a neutral config.yaml holding training parameters
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
        config_path: str,
        aws_config: dict[str, str],
        model_name: str,
        description: str,
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
        transformers_version: str = "4.56",
        pytorch_version: str = "2.8",
        py_version: str = "py312",
    ):
        super().__init__(
            schema,
            prompt_builder,
            training_batch_names,
            config_path,
            aws_config,
            model_name,
            description,
        )
        self.hyperparameters = self.config.to_hf_hyperparameters()
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.transformers_version = transformers_version
        self.pytorch_version = pytorch_version
        self.py_version = py_version

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

        train_jsonl = self._prepare_training_data("train.jsonl")

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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            base, source_folder, autocast_adapter_dtype=False
        )
        merged = model.merge_and_unload()
        merged.save_pretrained(target_folder, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        tokenizer.save_pretrained(target_folder)
        return True

    def post_process(
        self,
        model_card: ModelCard,
        s3_output_path: str | None,
        job_name: str | None,
        push_public: bool = False,
    ) -> None:
        """Download, merge and upload fine-tuned model.

        The primary publish target is the build bucket (unpacked, under
        models/{model_name}/{model_name}_{v}/)

        Set push_public=True to also push tarball to the public bucket.

        Args:
            model_card (ModelCard): Model card metadata.
            s3_output_path (str | None): S3 output path. If None, uses self.s3_output_path.
            job_name (str | None): SageMaker job name. If None, uses self.last_job_name.
            push_public (bool): Also upload the public tarball. Defaults to False.

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
        self._publish(str(target_folder), model_card, push_public)
