"""
batch_generator.py

Class to handle use of AWS batch inference API
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from itertools import islice
from pathlib import Path

from pydantic import BaseModel

from datagen.config import Config
from datagen.document_loader import DocumentLoader
from datagen.extraction import save_training_sample
from datagen.version_detector import get_schema_version
from mesa_types import Document
from utils.aws import AWS
from utils.llm import BatchOutputs


class BatchState(BaseModel):
    """State carried between batch generation runs.

    Attributes:
        job_id (str): Id of the most recently submitted batch job
        sent_record_ids (list[str]): Record ids of every document sent in a
            batch so far, so that later runs continue with unsent documents.
            Defaults to an empty list, which sends every document afresh

    """

    job_id: str
    sent_record_ids: list[str] = []


class BedrockBatchGenerator:
    """Bedrock batch sample generator for large-scale processing.

    Uses AWS Bedrock batch inference API for cost-effective large-scale generation.

    Args:
        system_prompt: System prompt for sample generation
        schema: Pydantic schema class for validation
        schema_name: Schema name for output filenames
        model_name: Model name from config.json (e.g., 'sonnet4', 'opus4')
        document_batches: S3 batch filenames to download
    """

    def __init__(
        self,
        system_prompt: str,
        schema: type[BaseModel],
        schema_name: str,
        model_name: str,
        document_batches: list[str],
    ):
        self.__config: Config = Config()
        self._logger: logging.Logger = logging.getLogger(__name__)
        if not self._logger.hasHandlers():
            self._logger.addHandler(logging.StreamHandler())

        self.__system_prompt: str = system_prompt
        self.__schema: type[BaseModel] = schema
        self.__schema_name: str = schema_name
        self.__schema_version: str = get_schema_version(schema_name)

        self.__model_id: str = self.__config.models[model_name].model
        self.__model_region: str = self.__config.models[model_name].region
        os.environ["AWS_REGION_NAME"] = self.__model_region
        self.__model_batch_file: str = self.__config.models[model_name].batch_file
        self.__output_folder_name: str = "./data/trainingdata/"

        # download and extract batches from S3
        batches: list[list[Path]] = []
        for batch_filename in document_batches:
            batch_name = batch_filename.replace(".tar.gz", "").replace(".tar", "")
            output_folder = Path(f"./data/documents/{batch_name}")
            self._logger.info(f"Downloading batch: {batch_filename}")
            DocumentLoader.download_and_extract(
                filename=batch_filename,
                output_folder=output_folder,
            )
            batches.append(sorted(output_folder.glob("*.json")))
        self.__document_files: list[Path] = DocumentLoader.interleave(batches)

    def get_document_files_count(self) -> int:
        return len(self.__document_files)

    @staticmethod
    def _record_id(doc: Document) -> str:
        return hashlib.md5(doc.content.encode()).hexdigest()

    def _documents_by_record_id(self) -> dict[str, Document]:
        return {
            BedrockBatchGenerator._record_id(doc): doc
            for doc in (
                Document.model_validate_json(doc_path.read_text())
                for doc_path in self.__document_files
            )
        }

    def _read_state(self) -> BatchState | None:
        if not Path(self.__config.job_id_file).exists():
            return None
        with open(self.__config.job_id_file) as state_file:
            return BatchState.model_validate_json(state_file.read())

    def _write_state(self, state: BatchState) -> None:
        with open(self.__config.job_id_file, "w") as state_file:
            state_file.write(state.model_dump_json())

    def _pending_documents(self, sample_size: int) -> dict[str, Document]:
        sent: set[str] = (
            set(state.sent_record_ids) if (state := self._read_state()) else set()
        )
        pending: dict[str, Document] = {
            record_id: doc
            for record_id, doc in self._documents_by_record_id().items()
            if record_id not in sent
        }
        if not pending:
            raise ValueError(
                f"All {len(sent)} available documents have already been sent in a batch"
            )
        if sample_size > len(pending):
            self._logger.warning(
                f"Requested {sample_size} samples but only {len(pending)} unsent documents "
                f"available. Will create {len(pending)} samples."
            )
        return dict(islice(pending.items(), sample_size))

    def _generate_batch(
        self, sample_size: int, file_name: str = "anthropic_batch_job.jsonl"
    ) -> list[str]:
        """Generate batch request file for Anthropic Bedrock model.

        Args:
            sample_size: Maximum number of samples to be generated, drawn in
                order from the documents not sent in a previous batch
            file_name: Output filename for batch request

        Returns:
            The record ids of the documents written to the batch request file

        """
        pending: dict[str, Document] = self._pending_documents(sample_size)
        with open(file_name, "w") as outfile:
            for record_id, doc in pending.items():
                print(
                    json.dumps(
                        AWS.create_anthropic_bedrock_batch_entry(
                            record_id,
                            self.__system_prompt,
                            doc.content,
                        )
                    ),
                    file=outfile,
                )

        self._logger.info(f"Generated batch file with {len(pending)} entries")
        return list(pending)

    def generate_via_batch(
        self,
        sample_size: int,
        bucket: str,
        bedrock_execution_role: str,
    ) -> str:
        """Generate samples via batch inference.

        Args:
            sample_size: Maximum number of samples to be generated
            bucket: The name of the bucket to which the batch
                specification should be uploaded
            bedrock_execution_role: The ARN of an IAM role with
                permissions to access S3 for batch specification and
                access cross-region models

        Returns:
            The id of the started job

        Raises:
            ValueError: If every available document has already been sent

        """
        previous: BatchState | None = self._read_state()
        # Create batch instruction JSONL file
        record_ids: list[str] = self._generate_batch(sample_size)
        job_id: str = "datagen/" + datetime.now().strftime("%Y-%m-%d-%H%M")
        AWS.run_batch_inference(
            job_id,
            self.__model_id,
            self.__model_batch_file,
            bucket,
            bedrock_execution_role,
            self.__model_region,
        )
        self._write_state(
            BatchState(
                job_id=job_id,
                sent_record_ids=(previous.sent_record_ids if previous else [])
                + record_ids,
            )
        )
        return job_id

    def __resolve_job_id(self) -> str | None:
        return state.job_id if (state := self._read_state()) else None

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
        batch_outputs: BatchOutputs
        job_id: str | None = self.__resolve_job_id() if bucket is not None else None
        if bucket is not None and job_id is not None:
            batch_outputs = AWS.get_batch_inference_outputs(
                self.__model_region, bucket, job_id, file_name
            )
        else:
            batch_outputs = AWS.parse_batch_output(file_name)

        docs: dict[str, Document] = self._documents_by_record_id()

        os.makedirs(self.__output_folder_name, exist_ok=True)
        successful_generations: int = 0
        failed_generations: int = 0
        for bedrock_batch_output in batch_outputs.outputs:
            try:
                if (doc := docs.get(bedrock_batch_output.recordId)) is None:
                    raise ValueError(
                        f"no document matches record {bedrock_batch_output.recordId}"
                    )

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
                self._logger.error(
                    f"Error processing batch output {bedrock_batch_output.recordId}: {e}"
                )
        self._logger.info(
            f"Processing complete: {successful_generations} successful, {failed_generations} failed"
        )
        return successful_generations, failed_generations

    def check_batch_output_status(self, bucket: str) -> bool:
        """Check whether a submitted Bedrock batch job's output exists in S3.

        Args:
            bucket: S3 bucket the Bedrock batch job writes its output to

        Returns:
            Whether an object under the job's output prefix ending in the
            expected batch output filename exists

        Raises:
            ValueError: If no batch job has been submitted yet

        """
        job_id: str | None = self.__resolve_job_id()
        if job_id is None:
            raise ValueError(
                "No batch job id found; generate_via_batch must be called first."
            )

        return any(
            object["Key"].endswith(self.__model_batch_file + ".out")
            for object in AWS.list_s3_objects(
                self.__model_region, bucket, job_id + "/output/"
            )
        )
