"""
llm_generator.py

Class to handle real-time APIs via LiteLLM
"""

import json
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from litellm import ModelResponse
from pydantic import BaseModel

from datagen import extraction
from datagen.document_loader import DocumentBatchLoader
from mesa_types import Document
from utils.llm import LLM


class LLMGenerator:
    """Real-time sample generator using any LLM provider.

    Supports any model accessible via LiteLLM (OpenAI, Anthropic, Bedrock, local models, etc.)

    Args:
        system_prompt: System prompt for sample generation
        user_prompt_function: Function to generate user prompt from document dict
        schema: Pydantic schema class for validation
        schema_name: Schema name for output filenames
        model_name: LiteLLM model string (e.g., 'gpt-4', 'claude-3-opus', 'bedrock/...')
        api_key: API key for the model provider
        document_batches: S3 batch filenames to download

    """

    def __init__(
        self,
        system_prompt: str,
        user_prompt_function: Callable[[dict[str, Any]], str],
        schema: type[BaseModel],
        schema_name: str,
        model_name: str,
        api_key: str,
        document_batches: list[str],
    ):
        self.__logger: logging.Logger = logging.getLogger(__name__)
        if not self.__logger.hasHandlers():
            self.__logger.addHandler(logging.StreamHandler())

        self.__system_prompt: str = system_prompt
        self.__user_prompt_function: Callable[[dict[str, Any]], str] = (
            user_prompt_function
        )
        self.__schema: type[BaseModel] = schema
        self.__schema_name: str = schema_name
        self.__model_name: str = model_name
        self.__api_key: str = api_key

        # auto-detect version from installed package
        pypi_package = f"londonaicentre-{schema_name}"
        try:
            raw_version = version(pypi_package)
            # convert e.g. 1.2.3 -> "1_2_3"
            version_parts = raw_version.split(".")[:3]
            self.__schema_version = "_".join(version_parts)
        except PackageNotFoundError as e:
            raise RuntimeError(
                f"Schema package '{pypi_package}' not found. "
            ) from e

        self.__output_folder_name: str = "./data/trainingdata/"

        # batches from S3
        self.__document_files: list[Path] = []
        for batch_filename in document_batches:
            batch_name = batch_filename.replace(".tar.gz", "").replace(".tar", "")
            output_folder = Path(f"./data/documents/{batch_name}")
            self.__logger.info(f"Downloading batch: {batch_filename}")
            DocumentBatchLoader.download_and_extract(
                filename=batch_filename,
                output_folder=output_folder,
            )
            self.__document_files.extend(sorted(output_folder.glob("document_*.json")))

    def _generate_sample(self, doc_path: Path) -> bool:
        """Generate structured output from a document.

        Args:
            doc_path: Path to document JSON file

        Returns:
            Whether sample generation is successful

        """
        try:
            doc = Document(**json.loads(doc_path.read_text()))
            user_prompt: str = self.__user_prompt_function(doc.model_dump())

            message: ModelResponse | None = LLM.completion(
                self.__model_name,
                self.__system_prompt,
                user_prompt,
                self.__api_key,
            )
            if message is None:
                return False

            content: str | None = message.choices[0].message.content
            if content is not None:
                return extraction.save_training_sample(
                    content,
                    doc.source,
                    doc.content,
                    self.__schema,
                    self.__schema_name,
                    self.__schema_version,
                    self.__output_folder_name,
                )
            else:
                return False
        except Exception as e:
            self.__logger.error(f"Error processing document {doc_path.name}: {e}")
            return False

    def generate(self, sample_size: int) -> None:
        """Generate samples via individual LLM inference calls.

        Args:
            sample_size: Number of samples to generate

        """
        os.makedirs(self.__output_folder_name, exist_ok=True)
        max_samples = len(self.__document_files)
        if sample_size > max_samples:
            self.__logger.warning(
                f"Requested {sample_size} samples but only {max_samples} documents available. "
                f"Will generate {max_samples} samples."
            )
            sample_size = max_samples

        successful_generations = 0
        failed_generations = 0
        processed = 0

        for doc_path in self.__document_files:
            if processed >= sample_size:
                break

            # load to check if exists
            try:
                doc = Document(**json.loads(doc_path.read_text()))
                output_filename = os.path.join(
                    self.__output_folder_name,
                    extraction.get_output_filename(
                        self.__schema_name, self.__schema_version, doc
                    ),
                )

                # skip if exists
                if os.path.exists(output_filename):
                    self.__logger.info(
                        f"Output already exists for {doc_path.name}, skipping"
                    )
                    processed += 1
                    continue

                # generate
                if self._generate_sample(doc_path):
                    successful_generations += 1
                else:
                    failed_generations += 1

                processed += 1

            except Exception as e:
                self.__logger.error(f"Error checking document {doc_path.name}: {e}")
                failed_generations += 1
                processed += 1

        self.__logger.info(
            f"Processing complete: {successful_generations} successful, {failed_generations} failed"
        )

    def run_backfill(self) -> None:
        """Process documents that don't have output files yet."""
        os.makedirs(self.__output_folder_name, exist_ok=True)

        successful_generations = 0
        failed_generations = 0

        for doc_path in self.__document_files:
            try:
                # load to check if exists
                doc = Document(**json.loads(doc_path.read_text()))
                output_filename = os.path.join(
                    self.__output_folder_name,
                    extraction.get_output_filename(
                        self.__schema_name, self.__schema_version, doc
                    ),
                )

                # skip if exists
                if os.path.exists(output_filename):
                    continue

                # generate
                self.__logger.info(f"Backfilling {doc_path.name}")
                if self._generate_sample(doc_path):
                    successful_generations += 1
                else:
                    failed_generations += 1

            except Exception as e:
                self.__logger.error(f"Error processing document {doc_path.name}: {e}")
                failed_generations += 1

        if successful_generations == 0 and failed_generations == 0:
            self.__logger.info("No missing samples to backfill")
        else:
            self.__logger.info(
                f"Backfill complete: {successful_generations} successful, {failed_generations} failed"
            )
