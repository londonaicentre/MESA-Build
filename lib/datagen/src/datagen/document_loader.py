"""
document_loader.py

Utilites to download document batches from S3
Before passing into training data generation
"""

from pathlib import Path
import json
import re
import tarfile
from typing import Literal
from mesa_types import Document
from utils.aws import AWS


class DocumentLoader:
    """
    Download document batches from S3, extract, and validate
    """

    @staticmethod
    def download_and_extract(
        filename: str,
        output_folder: Path,
        bucket: str = "aicentre-nlpteam-mesa-build",
        s3_prefix: str = "documents",
        region: str = "eu-west-2",
    ) -> int:
        """
        Download document batch from S3, extract to folder, and validate.

        Args:
            filename:
                Batch filename (e.g., "cancer-batch-2025-12-31-001.tar.gz")
            output_folder:
                Local folder to extract to
            bucket:
                S3 bucket name (default: "aicentre-nlpteam-mesa-build")
            s3_prefix:
                S3 folder (default: "documents")
            region:
                AWS region (default: "eu-west-2")

        Returns:
            Count of documents extracted and validated
        """
        # S3 -> local cache dir
        cache_dir = Path("data/_cache/document_batches")
        cache_dir.mkdir(parents=True, exist_ok=True)
        tar_path = cache_dir / filename
        s3_key = f"{s3_prefix}/{filename}" if s3_prefix else filename

        # do not redownload
        if not tar_path.exists():
            success = AWS.download_file(
                region_name=region,
                bucket=bucket,
                file_name=str(tar_path),
                object_name=s3_key,
            )

            if not success:
                raise Exception(f"Failed to download s3://{bucket}/{s3_key}")

        # always re-extract
        output_folder.mkdir(parents=True, exist_ok=True)
        mode: Literal["r", "w", "r:gz", "w:gz"] = (
            "r:gz" if tar_path.suffix == ".gz" else "r"
        )
        doc_count = 0

        with tarfile.open(tar_path, mode) as tar:
            for member in tar.getmembers():
                # legacy documents have basename "document_*"
                basename = member.name.split("/")[-1]
                is_legacy_document = basename.startswith(
                    "document_"
                ) and basename.endswith(".json")
                is_hashed_document = bool(
                    re.compile(r"^[0-9a-f]{32}\.json$").match(basename)
                )
                if is_legacy_document or is_hashed_document:
                    file = tar.extractfile(member)
                    if file:
                        doc = Document.model_validate_json(file.read())

                        output_path = output_folder / basename
                        output_path.write_text(json.dumps(doc.model_dump(), indent=2))
                        doc_count += 1

        return doc_count

    @staticmethod
    def list_available_document_batches(
        bucket: str = "aicentre-nlpteam-mesa-build",
        s3_prefix: str = "documents",
        region: str = "eu-west-2",
    ) -> list[str]:
        """
        List document batch filenames available in S3, suitable for
        passing as `filename` to `download_and_extract`.

        Args:
            bucket:
                S3 bucket name (default: "aicentre-nlpteam-mesa-build")
            s3_prefix:
                S3 folder (default: "documents")
            region:
                AWS region (default: "eu-west-2")

        Returns:
            List of batch filenames (e.g., "cancer-batch-2025-12-31-001.tar.gz")

        """
        prefix = f"{s3_prefix}/" if s3_prefix else ""
        return [
            filename
            for object in AWS.list_s3_objects(region, bucket, prefix)
            if (filename := str(object["Key"]).removeprefix(prefix)).endswith(
                (".tar", ".tar.gz")
            )
        ]
