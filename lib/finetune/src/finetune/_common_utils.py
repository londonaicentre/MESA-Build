"""
_common_utils.py

Common utilities shared by hf_estimator.py and mlx_trainer.py
- job id creation
- model card construction
- archive (tar) and upload of models
"""

import io
import tarfile
from datetime import datetime
from pathlib import Path

from mesa_types.model_card import ModelCard
from pydantic import BaseModel

from utils.aws import AWS


def make_job_id(description: str) -> str:
    """Build a timestamped job ID.

    Args:
        description: Job description (sagemaker dislikes underscores).

    Returns:
        ``f"{YYYYmmdd-HHMMSS}-{description}"``.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{description}"


def build_model_card(
    *,
    base_model: str,
    model_name: str,
    major: int,
    minor: int,
    patch: int,
    model_description: str,
    training_data: list[str],
    output_schema: type[BaseModel],
) -> ModelCard:
    """Build a ``ModelCard`` from training metadata.

    The per-trainer differences are the caller's responsibility: HF passes its
    ``base_model`` + the S3 input path as ``training_data``; MLX passes its ``base_model``
    + the training batch names.

    Args:
        base_model: HF identifier of the base model.
        model_name: Model name used for the card and uploaded archive.
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.
        model_description: Human-readable model description.
        training_data: References to the training data (S3 path or batch names).
        output_schema: Pydantic schema describing the model output.

    Returns:
        A populated ``ModelCard``.
    """
    return ModelCard(
        base_model_hf=base_model,
        model_name=model_name,
        major=major,
        minor=minor,
        patch=patch,
        model_description=model_description,
        training_data=training_data,
        output_schema=output_schema,
    )


def archive_and_upload(
    *,
    target_folder: str,
    model_card: ModelCard,
    model_name: str,
    region: str,
    bucket: str = "aicentre-nlpteam-mesa-public",
) -> bool:
    """Tar the merged model + ``model_card.yml`` + ``LICENSE.md`` and upload to S3.

    Args:
        target_folder: Folder containing the merged/fused model.
        model_card: Model card metadata (tarred in as ``model_card.yml``).
        model_name: S3 key prefix the archive is uploaded under.
        region: AWS region.
        bucket: S3 bucket name. Defaults to 'aicentre-nlpteam-mesa-public'.

    Returns:
        True if upload successful.

    Raises:
        ValueError: if the upload fails.
    """
    target_path = Path(target_folder)
    archive_name = f"{model_card.model_name}_{model_card.major}_{model_card.minor}_{model_card.patch}.tar.gz"
    archive_path = target_path.parent / archive_name
    if not archive_path.exists():
        with tarfile.open(archive_path, "w:gz") as tar:
            for item in target_path.iterdir():
                tar.add(item, arcname=item.name)
            yaml_bytes: bytes = model_card.to_yaml_bytes()
            tarinfo: tarfile.TarInfo = tarfile.TarInfo(name="model_card.yml")
            tarinfo.size = len(yaml_bytes)
            tar.addfile(tarinfo, io.BytesIO(yaml_bytes))
            tar.add(Path(__file__).parents[2] / "LICENSE.md", arcname="LICENSE.md")
    success = AWS.upload_file(
        region_name=region,
        file_name=str(archive_path),
        bucket=bucket,
        object_name=archive_name,
        path=model_name,
    )
    if not success:
        raise ValueError("Failed to upload merged model weights")
    return True
