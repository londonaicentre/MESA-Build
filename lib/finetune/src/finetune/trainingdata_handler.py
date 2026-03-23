"""
trainingdata_handler.py

Utilities to download/prepare training data from S3 before fine-tuning
Expects training data to be in OpenAI JSONL format
Can take multiple JSONL files and concat them
"""

import json
import logging
import random
from pathlib import Path

from pydantic import BaseModel

from mesa_types import TrainingSample
from utils.aws import AWS

logger = logging.getLogger(__name__)


class TrainingDataHandler:
    """
    Download and prepare training data batches from S3
    """

    @staticmethod
    def prepare(
        schema: type[BaseModel],
        system_prompt: str,
        training_batch_names: list[str],
        bucket: str = "aicentre-nlpteam-mesa-build",
        s3_prefix: str = "trainingdata",
        output_file: str = "train.jsonl",
        region: str = "eu-west-2",
        shuffle: bool = True,
    ) -> str:
        """Download, validate, and combine training data from S3.

        Args:
            schema: Pydantic schema class for validation (e.g., OncologyModel)
            system_prompt: Expected system prompt (from PromptBuilder.build_main_prompt())
            training_batch_names: List of training batch folder names
                (e.g., ["20260123-094248_test-batch"])
            bucket: S3 bucket name
            s3_prefix: S3 prefix path
            output_file: Local output filename
            region: AWS region
            shuffle: Whether to shuffle combined samples

        Returns:
            Path to local combined JSONL file
        """
        cache_dir = Path("data/_cache/training_batches")
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_samples = []

        for batch_name in training_batch_names:
            # each batch folder expected to contain exactly 1 JSONL file
            s3_valid_prefix = f"{s3_prefix}/{batch_name}"

            objects = AWS.list_s3_objects(
                region_name=region,
                bucket=bucket,
                prefix=s3_valid_prefix + "/",
            )

            jsonl_files = [
                obj["Key"] for obj in objects if obj["Key"].endswith(".jsonl")
            ]

            if len(jsonl_files) == 0:
                raise ValueError(f"No JSONL file found in {batch_name}")
            elif len(jsonl_files) > 1:
                raise ValueError(
                    f"Multiple JSONL files found in {batch_name}: {jsonl_files}"
                )

            jsonl_key = jsonl_files[0]

            # download to cache
            jsonl_path = cache_dir / f"{batch_name}.jsonl"

            if not jsonl_path.exists():
                logger.info(f"Downloading {jsonl_key}...")
                success = AWS.download_file(
                    region_name=region,
                    bucket=bucket,
                    file_name=str(jsonl_path),
                    object_name=jsonl_key,
                )
                if not success:
                    raise ValueError(f"Failed to download {jsonl_key}")

            # validation
            logger.info(f"Validating {jsonl_path}...")
            valid_samples = 0
            invalid_samples = 0

            with open(jsonl_path) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = TrainingSample.model_validate(json.loads(line))

                        # vs expected system prompt
                        sample_system_prompt = sample.messages[0].content
                        if sample_system_prompt.replace("\n", "").replace(
                            " ", ""
                        ) != system_prompt.replace("\n", "").replace(" ", ""):
                            raise ValueError("System prompt mismatch")

                        # vs schema
                        assistant_content = sample.messages[2].content
                        json_str = (
                            assistant_content.replace("<output>", "")
                            .replace("</output>", "")
                            .strip()
                        )
                        output_data = json.loads(json_str)
                        schema.model_validate(output_data)

                        all_samples.append(sample)
                        valid_samples += 1

                    except Exception:
                        logger.warning(f"Line {line_num}: Invalid, skipping")
                        invalid_samples += 1

            logger.info(
                f"Loaded {valid_samples} valid samples from {batch_name} "
                f"({invalid_samples} invalid)"
            )

        if not all_samples:
            raise ValueError("No valid training samples found")

        if shuffle:
            random.shuffle(all_samples)
            logger.info("Shuffled training samples")

        output_path = Path(output_file)
        with open(output_path, "w") as f:
            for sample in all_samples:
                f.write(sample.model_dump_json() + "\n")

        logger.info(f"Prepared {len(all_samples)} total samples in {output_path}")

        return str(output_path)
