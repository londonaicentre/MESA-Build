import hashlib
import json
import logging
import os
import re
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any, Callable, cast

import litellm
from litellm import Choices, ModelResponse
from pydantic import BaseModel

from datagen.config import Config
from datagen.document_loader import DocumentBatchLoader
from mesa_types import Document
from utils.aws import AWS
from utils.llm import LLM, BatchOutputs

litellm.suppress_debug_info = (
    True  # suppress unhelpful library output on rate limit error
)


class SampleGenerator:
    """Synthetic data generator for fine-tuning LLMs.

    Args:
        system_prompt: System prompt for sample generation
        user_prompt_function: User prompt generation function
        schema: Schema class to use for validation
        schema_name: Schema name for output filenames
        model_name: Name of model to use on AWS Bedrock
        document_batches: S3 batch filenames to download
        bedrock_api_key: API key to access AWS Bedrock

    """

    def __init__(
        self,
        system_prompt: str,
        user_prompt_function: Callable[[dict[str, Any]], str],
        schema: type[BaseModel],
        schema_name: str,
        model_name: str,
        document_batches: list[str],
        bedrock_api_key: str | None = None,
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

        # auto-detect version from installed package
        pypi_package = f"londonaicentre-{schema_name}"
        try:
            raw_version = version(pypi_package)  # e.g., "2.0.0"
            # convert e.g. 1.2.3 -> "123"
            version_parts = raw_version.split(".")[:3]
            self.__schema_version = "".join(v.zfill(1) for v in version_parts)
        except PackageNotFoundError:
            self.__logger.warning(f"Package {pypi_package} not found, using version '000'")
            self.__schema_version = "000"

        self.__model_id: str = self.__config.models[model_name].model
        self.__model_region: str = self.__config.models[model_name].region
        os.environ["AWS_REGION_NAME"] = self.__model_region
        self.__model_batch_file: str = self.__config.models[model_name].batch_file
        self.__bedrock_api_key: str | None = bedrock_api_key
        self.__output_folder_name: str = "./data/trainingdata/"

        # download and extract batches from S3
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

    def _get_output_filename(self, doc: Document) -> str:
        """Generate metadata-embedded output filename for a document.

        Args:
            doc: Document to generate filename for

        Returns:
            Filename in format: {schema_name}{version}_{source}_{content_hash}.json

        """
        content_hash = hashlib.md5(doc.content.encode()).hexdigest()[:8]
        return f"{self.__schema_name}{self.__schema_version}_{doc.source}_{content_hash}.json"

    def _extract_json_from_response(self, response: str) -> dict[str, Any] | None:
        """Extract JSON from Claude's response

        Args:
            response (str): Claude's response

        Returns:
            JSON if parse successful, None otherwise

        """

        # try to parse response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # try to find JSON in a code block
            extracted: bool
            content: str
            extracted, _, content = LLM.extract_output_content(response)
            if extracted:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass

            # if fails, try to find any JSON-like structure
            json_match = re.search(r"{[\s\S]*}", response)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    raise json.JSONDecodeError(
                        "Could not parse JSON from response", "", 0
                    )
            self.__logger.error("No valid JSON found in response")
            return None

    def _validate_with_pydantic(
        self, output_data: dict[str, Any]
    ) -> tuple[bool, BaseModel | None]:
        """Validate the output against the Pydantic schema

        Args:
            output_data (dict): JSON data to validate

        Returns:
            True, Parsed JSON if validation successful, False, None otherwise.

        """
        try:
            validated_report: BaseModel = self.__schema(**output_data)
            return True, validated_report
        except Exception as e:
            self.__logger.error(f"Pydantic validation error: {e}")
            return False, None

    def _extract_validate_and_save_sample(
        self, response: str, source: str, content: str
    ) -> bool:
        """Extract sample from model response, validate it and save it.

        Args:
            response: The raw response from the model
            source: Document source for filename
            content: Document content for filename hash

        Returns:
            Whether the operations were successful

        """
        json_output: dict[str, Any] | None = self._extract_json_from_response(response)
        if json_output is not None:
            if (
                not isinstance(json_output, dict)
                or "content" not in json_output
                or "output" not in json_output
            ):
                self.__logger.error("Invalid schema format in output")
                return False

            # validate against schema
            is_valid: bool
            validated_output: BaseModel | None
            is_valid, validated_output = self._validate_with_pydantic(
                json_output["output"]
            )
            if is_valid and validated_output is not None:
                # convert pydantic model to dict for json serialization
                json_output["output"] = validated_output.model_dump()

                doc = Document(content=content, source=source, timestamp="")
                output_filename: str = os.path.join(
                    self.__output_folder_name, self._get_output_filename(doc)
                )
                try:
                    with open(output_filename, "w", encoding="utf-8") as f:
                        json.dump(json_output, f, indent=4, ensure_ascii=False)
                    self.__logger.info(
                        f"Successfully saved output to {output_filename}"
                    )
                    return True
                except Exception as e:
                    self.__logger.error(f"Error saving JSON to file: {e}")
                    return False
            else:
                self.__logger.error("Pydantic validation failed")

                # for debugging later
                doc = Document(content=content, source=source, timestamp="")
                debug_filename: str = os.path.join(
                    self.__output_folder_name,
                    f"invalid_{self._get_output_filename(doc)}",
                )
                with open(debug_filename, "w", encoding="utf-8") as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)
                return False
        else:
            self.__logger.warning("Skipping file save due to JSON parsing failure")
            self.__logger.debug(
                "Claude response:",
                response,
            )
            return False

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
            if self.__bedrock_api_key is None:
                return False
            message: ModelResponse | None = AWS.bedrock_completion(
                self.__model_id,
                self.__system_prompt,
                user_prompt,
                self.__bedrock_api_key,
            )
            if message is None:
                return False
            content: str | None = cast(Choices, message.choices[0]).message.content
            if content is not None:
                return self._extract_validate_and_save_sample(
                    content, doc.source, doc.content
                )
            else:
                return False
        except Exception as e:
            self.__logger.error(f"Error processing document {doc_path.name}: {e}")
            return False

    # real-time generation

    def generate(self, sample_size: int) -> None:
        """Generate samples via individual AWS Bedrock inference calls.

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
                    self.__output_folder_name, self._get_output_filename(doc)
                )

                # skip if exists
                if os.path.exists(output_filename):
                    self.__logger.info(f"Output already exists for {doc_path.name}, skipping")
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

    # backfill

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
                    self.__output_folder_name, self._get_output_filename(doc)
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

    # batch

    def _generate_batch(
        self, sample_size: int, file_name: str = "anthropic_batch_job.jsonl"
    ) -> str:
        """Generate batch request file for Anthropic model.

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
                doc = Document(**json.loads(doc_path.read_text()))
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
        """Generate samples via batch inference

        Args:
            sample_size (int): Number of samples to be generated
            bucket (str): The name of the bucket to which the batch
                specification should be uploaded
            bedrock_execution_role (str): The ARN of an IAM role with
                permissions to access S3 for batch specification and
                access cross-region models

        Returns:
            str: the id of the started job

        """

        # Process all samples from bootstrap file in batch mode
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
            bucket (str, optional): The bucket from which the batch
                sample outputs file should be downloaded if it is not local
            file_name (str, optional): Batch output file from which
                to extract samples (default to `anthropic_batch_job.jsonl.out`)

        Returns:
            tuple: the number of successfully and unsuccessfully parsed files

        """
        if bucket is not None:
            if Path(self.__config.job_id_file).exists():
                with open(self.__config.job_id_file) as job_id_file:
                    path: str = json.loads(job_id_file.read())["job_id"] + "/output/*"
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
                    doc = Document(**json.loads(doc_path.read_text()))

                    if self._extract_validate_and_save_sample(
                        str(bedrock_batch_output.modelOutput.content[0].text),
                        doc.source,
                        doc.content,
                    ):
                        successful_generations += 1
                    else:
                        failed_generations += 1
                except Exception as e:
                    failed_generations += 1
                    self.__logger.error(f"Error processing batch output {sample_id}: {e}")
        self.__logger.info(
            f"Processing complete: {successful_generations} successful, {failed_generations} failed"
        )
        return successful_generations, failed_generations
