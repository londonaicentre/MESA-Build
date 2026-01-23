"""
training_upload.py

Upload training data batches to S3
"""

import json
import logging
import os
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from datagen.version_detector import get_schema_version
from mesa_types import TrainingExample
from utils.aws import AWS

logger = logging.getLogger(__name__)


class TrainingDataUploader:
    """Upload training data batches to S3"""

    @staticmethod
    def _create_jsonl(
        samples: list[dict[str, Any]],
        system_prompt: str,
        short_description: str,
    ) -> str:
        """Create OpenAI messages format JSONL file.

        Args:
            samples: List of training samples (dicts with 'content' and 'output')
            system_prompt: System prompt string
            short_description: Short description for filename

        Returns:
            Filename of JSONL file
        """
        filename = f"train_openai_{short_description}.jsonl"
        with open(filename, "w") as f:
            for sample in samples:
                assistant_response = (
                    f"<output>\n{json.dumps(sample['output'], indent=2)}\n</output>"
                )
                message = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": sample["content"]},
                        {"role": "assistant", "content": assistant_response},
                    ]
                }
                f.write(json.dumps(message) + "\n")
        return filename

    @staticmethod
    def _create_metadata(
        schema_name: str,
        schema_version: str,
        short_description: str,
        long_description: str,
        num_samples: int,
    ) -> str:
        """Create metadata YAML file.

        Args:
            schema_name: Schema package name
            schema_version: Schema version string
            short_description: Short description
            long_description: Long description
            num_samples: Number of samples

        Returns:
            Filename of created YAML file
        """
        filename = "metadata.yaml"
        metadata = {
            "schema": schema_name,
            "schema_version": schema_version,
            "short_description": short_description,
            "description": long_description,
            "num_samples": num_samples,
            "created_at": datetime.now().isoformat(),
        }
        with open(filename, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)
        return filename

    @staticmethod
    def _create_samples_archive(
        valid_files: list[Path],
        short_description: str,
    ) -> str:
        """Create tar.gz archive of valid training samples.

        Args:
            valid_files: List of valid JSON file paths
            short_description: Short description for filename

        Returns:
            Filename of created tar.gz archive
        """
        filename = f"train_openai_{short_description}_samples.tar.gz"
        with tarfile.open(filename, "w:gz") as tar:
            for json_file in valid_files:
                tar.add(json_file, arcname=json_file.name)
        return filename

    @staticmethod
    def upload(
        schema: type[BaseModel],
        schema_name: str,
        system_prompt: str,
        short_description: str,
        long_description: str | None = None,
        input_folder: Path = Path("./data/trainingdata"),
        bucket: str = "aicentre-nlpteam-mesa-build",
        s3_prefix: str = "trainingdata",
        region: str = "eu-west-2",
    ) -> str:
        """Upload training data to S3.

        Args:
            schema: Pydantic schema class (e.g., OncologyModel)
            schema_name: Schema package name (e.g., "oncoschema")
            system_prompt: System prompt string from PromptBuilder
            short_description: Short description for folder/file names
            long_description: Long description for metadata (optional)
            input_folder: Local folder with training JSONs
            bucket: S3 bucket name
            s3_prefix: S3 prefix path
            region: AWS region

        Returns:
            S3 URI of uploaded batch

        Raises:
            ValueError: If validation fails or no valid samples found
        """
        schema_version = get_schema_version(schema_name)

        if not input_folder.exists():
            raise ValueError(f"Input folder does not exist: {input_folder}")

        # collect samples after validation
        samples = []
        valid_files = []

        for json_file in input_folder.glob("*.json"):
            try:
                example = TrainingExample.model_validate_json(json_file.read_text())
                schema.model_validate(example.output)
                samples.append({"content": example.content, "output": example.output})
                valid_files.append(json_file)
            except ValidationError as e:
                logger.warning(f"Skipping invalid sample {json_file.name}: {e}")
            except Exception as e:
                logger.warning(f"Could not load {json_file.name}: {e}")

        if not samples:
            raise ValueError(f"No valid samples found in {input_folder}")

        logger.info(f"Loaded {len(samples)} valid training samples")

        # create S3 path
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}_{short_description}"
        s3_run_path = f"{s3_prefix}/{run_name}"

        logger.info(f"Uploading to s3://{bucket}/{s3_run_path}/")

        # create, upload JSONL in openAI format
        jsonl_filename = TrainingDataUploader._create_jsonl(
            samples, system_prompt, short_description
        )
        AWS.upload_file(region, jsonl_filename, bucket, jsonl_filename, s3_run_path)
        # 1st filename is local file
        # 2nd filename is s3 object name
        os.remove(jsonl_filename)

        # create metadata
        metadata_filename = TrainingDataUploader._create_metadata(
            schema_name,
            schema_version,
            short_description,
            long_description or short_description,
            len(samples),
        )
        AWS.upload_file(
            region, metadata_filename, bucket, metadata_filename, s3_run_path
        )
        os.remove(metadata_filename)

        # create and upload samples archive
        archive_filename = TrainingDataUploader._create_samples_archive(
            valid_files, short_description
        )
        AWS.upload_file(region, archive_filename, bucket, archive_filename, s3_run_path)
        os.remove(archive_filename)

        s3_uri = f"s3://{bucket}/{s3_run_path}/"
        logger.info(f"Upload complete: {s3_uri}")

        return s3_uri
