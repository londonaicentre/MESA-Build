"""
batch_generator.py

Class to handle use of AWS batch inference API
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from datagen.config import Config
from datagen.document_loader import DocumentLoader
from datagen.extraction import save_training_sample
from datagen.version_detector import get_schema_version
from mesa_types import Document
from utils.aws import AWS
from utils.llm import BatchOutputs


class BedrockBatchGenerator:
    """Bedrock batch sample generator for large-scale processing.

    Uses AWS Bedrock batch inference API for cost-effective large-scale generation.

    Args:
        system_prompt: System prompt for sample generation
        user_prompt_function: Function to generate user prompt from document dict
        schema: Pydantic schema class for validation
        schema_name: Schema name for output filenames
        model_name: Model name from config.json (e.g., 'sonnet4', 'opus4')
        document_batches: S3 batch filenames to download
    """

    def __init__(
        self,
        system_prompt: str,
        user_prompt_function: Callable[[dict[str, Any]], str],
        schema: type[BaseModel],
        schema_name: str,
        model_name: str,
        document_batches: list[str],
    ):
        self.__config: Config = Config()
        self.__logger: logging.Logger = logging.getLogger(__name__)
        if not self.__logger.hasHandlers():
            self.__logger.addHandler(logging.StreamHandler())

        self.__system_prompt: str = system_prompt
        self.__user_prompt_function: Callable[[dict[str, Any]], str] = (
            user_prompt_function
        )
        self.__schema: type[BaseModel] = schema
        self.__schema_name: str = schema_name
        self.__schema_version: str = get_schema_version(schema_name)

        self.__model_id: str = self.__config.models[model_name].model
        self.__model_region: str = self.__config.models[model_name].region
        os.environ["AWS_REGION_NAME"] = self.__model_region
        self.__model_batch_file: str = self.__config.models[model_name].batch_file
        self.__output_folder_name: str = "./data/trainingdata/"

        # download and extract batches from S3
        self.__document_files: list[Path] = []
        for batch_filename in document_batches:
            batch_name = batch_filename.replace(".tar.gz", "").replace(".tar", "")
            output_folder = Path(f"./data/documents/{batch_name}")
            self.__logger.info(f"Downloading batch: {batch_filename}")
            DocumentLoader.download_and_extract(
                filename=batch_filename,
                output_folder=output_folder,
            )
            self.__document_files.extend(sorted(output_folder.glob("document_*.json")))

    def _generate_batch(
        self, sample_size: int, file_name: str = "anthropic_batch_job.jsonl"
    ) -> str:
        """Generate batch request file for Anthropic Bedrock model.

        Args:
            sample_size: Number of samples to be generated
            file_name: Output filename for batch request

        Returns:
            The batch request filename

        """
        max_samples = len(self.__document_files)
        if sample_size > max_samples:
            self.__logger.warning(
                f"Requested {sample_size} samples but only {max_samples} documents available. "
                f"Will create {max_samples} samples."
            )
            sample_size = max_samples

        with open(file_name, "w") as outfile:
            for idx, doc_path in enumerate(self.__document_files[:sample_size]):
                doc = Document.model_validate_json(doc_path.read_text())
                print(
                    json.dumps(
                        AWS.create_anthropic_bedrock_batch_entry(
                            str(idx),
                            self.__system_prompt,
                            self.__user_prompt_function(doc.model_dump()),
                        )
                    ),
                    file=outfile,
                )

        self.__logger.info(f"Generated batch file with {sample_size} entries")
        return file_name

    def generate_via_batch(
        self,
        sample_size: int,
        bucket: str,
        bedrock_execution_role: str,
    ) -> str:
        """Generate samples via batch inference.

        Args:
            sample_size: Number of samples to be generated
            bucket: The name of the bucket to which the batch
                specification should be uploaded
            bedrock_execution_role: The ARN of an IAM role with
                permissions to access S3 for batch specification and
                access cross-region models

        Returns:
            The id of the started job

        """
        # Create batch instruction JSONL file
        self._generate_batch(sample_size)
        job_id: str = "datagen/" + datetime.now().strftime("%Y-%m-%d-%H%M")
        AWS.run_batch_inference(
            job_id,
            self.__model_id,
            self.__model_batch_file,
            bucket,
            bedrock_execution_role,
            self.__model_region,
        )
        with open(self.__config.job_id_file, "w") as job_id_file:
            job_id_file.write(json.dumps({"job_id": job_id}))
        return job_id

    def extract_batch_output(
        self,
        bucket: str | None = None,
        file_name: str = "anthropic_batch_job.jsonl.out",
    ) -> tuple[int, int]:
        """Transform batch inference sample outputs to the same format
            (set of output files) as real-time generated samples.

        Args:
            bucket: The bucket from which the batch
                sample outputs file should be downloaded if it is not local
            file_name: Batch output file from which
                to extract samples (default to `anthropic_batch_job.jsonl.out`)

        Returns:
            Tuple of (successful_count, failed_count)

        """
        if bucket is not None:
            if Path(self.__config.job_id_file).exists():
                with open(self.__config.job_id_file) as job_id_file:
                    path: str = (
                        json.loads(job_id_file.read())["job_id"] + "/output/*"
                    )
                    if not AWS.download_file_with_wildcard(
                        self.__model_region,
                        bucket,
                        file_name,
                        file_name,
                        path,
                    ):
                        raise ValueError(
                            f"Error downloading file {file_name} from {bucket} at path {path}."
                        )
        os.makedirs(self.__output_folder_name, exist_ok=True)
        successful_generations: int = 0
        failed_generations: int = 0
        with open(file_name) as batch_output_file:
            for sample_id, bedrock_batch_output in enumerate(
                BatchOutputs.model_validate(
                    {"outputs": [json.loads(line) for line in batch_output_file]}
                ).outputs
            ):
                try:
                    doc_path = self.__document_files[sample_id]
                    doc = Document.model_validate_json(doc_path.read_text())

                    if save_training_sample(
                        str(bedrock_batch_output.modelOutput.content[0].text),
                        doc.source,
                        doc.content,
                        self.__schema,
                        self.__schema_name,
                        self.__schema_version,
                        self.__output_folder_name,
                    ):
                        successful_generations += 1
                    else:
                        failed_generations += 1
                except Exception as e:
                    failed_generations += 1
                    self.__logger.error(
                        f"Error processing batch output {sample_id}: {e}"
                    )
        self.__logger.info(
            f"Processing complete: {successful_generations} successful, {failed_generations} failed"
        )
        return successful_generations, failed_generations
