import io
import json
import logging
import tarfile
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Self

from mesa_types.model_card import ModelCard
from pydantic import BaseModel

from finetune.config import FinetuneConfig
from finetune.trainingdata_handler import TrainingDataHandler
from utils.aws import AWS
from utils.prompt import BasePromptBuilder

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Base orchestrator for LoRA fine-tuning.

    Args:
        schema: Pydantic schema class for validation.
        prompt_builder: Prompt builder instance.
        training_batch_names: List of S3 training batch folder names.
        config_path: Path to a neutral config.yaml holding training parameters.
        aws_config: AWS configuration dict with bucket, region and (optionally) role.
        model_name: Model name used for the model card and the uploaded archive.
        description: Job description, used for naming.
        config: Pre-loaded config, bypassing ``config_path``. Used when rebuilding a
            trainer from a serialised spec; defaults to loading ``config_path``.
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
        config: FinetuneConfig | None = None,
    ):
        self.schema = schema
        self.prompt_builder = prompt_builder
        self.training_batch_names = training_batch_names
        self.aws_config = aws_config
        self.model_name = model_name
        self.description = description

        self.config = config if config is not None else FinetuneConfig.load(config_path)
        self.base_model = self.config.training.base_model
        self.max_seq_length = self.config.training.max_seq_length

        # job ID (sagemaker does not like underscores!)
        self.job_id = LoRATrainer._make_job_id(description)

        # pass from an aws config dict (role unused for local training)
        self.bucket = aws_config["bucket"]
        self.region = aws_config["region"]
        self.role = aws_config.get("role", "")

    @staticmethod
    def _make_job_id(description: str) -> str:
        return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{description}"

    @staticmethod
    def _get_uploaded_model_folder_prefix(model_card: ModelCard) -> str:
        return f"models/{model_card.model_name}/{model_card.model_identifier}"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the trainer's state to a JSON-ready dict.

        Returns:
            dict[str, Any]: The trainer state, ready for ``json.dumps``.

        """
        return {
            "schema": {
                "module": self.schema.__module__,
                "qualname": self.schema.__qualname__,
            },
            "prompt_builder": {
                "module": type(self.prompt_builder).__module__,
                "qualname": type(self.prompt_builder).__qualname__,
            },
            "training_batch_names": self.training_batch_names,
            "config": self.config.model_dump(mode="json"),
            "aws_config": self.aws_config,
            "model_name": self.model_name,
            "description": self.description,
            "job_id": self.job_id,
        }

    def to_json(self) -> str:
        """Serialise the trainer to a single-line JSON string.

        Returns:
            str: Compact JSON suitable for passing between CI steps.

        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, source: str) -> Self:
        """Rebuild a trainer from a JSON spec produced by ``to_json``.

        Args:
            source (str): The JSON spec itself, or a path to a file containing it.
                Parsed as JSON first; if that fails, read as a file path.

        Returns:
            Self: A trainer of the calling class with its state restored.

        """
        try:
            data: dict[str, Any] = json.loads(source)
        except json.JSONDecodeError:
            data = json.loads(Path(source).read_text())
        trainer: Self = cls(**cls._constructor_kwargs(data))
        trainer._restore_runtime(data)
        return trainer

    @classmethod
    def _constructor_kwargs(cls, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": getattr(
                import_module(data["schema"]["module"]), data["schema"]["qualname"]
            ),
            "prompt_builder": getattr(
                import_module(data["prompt_builder"]["module"]),
                data["prompt_builder"]["qualname"],
            )(),
            "training_batch_names": data["training_batch_names"],
            "config_path": "",
            "aws_config": data["aws_config"],
            "model_name": data["model_name"],
            "description": data["description"],
            "config": FinetuneConfig.model_validate(data["config"]),
        }

    def _restore_runtime(self, data: dict[str, Any]) -> None:
        self.job_id = data["job_id"]

    def build_model_card(
        self,
        major: int,
        minor: int,
        patch: int,
        model_description: str | None = None,
    ) -> ModelCard:
        """Build a ``ModelCard`` from this trainer's state.

        Args:
            major (int): Major version number.
            minor (int): Minor version number.
            patch (int): Patch version number.
            model_description (str | None): Human-readable model description.
                Defaults to the trainer's ``description``.

        Returns:
            ModelCard: A populated model card.

        """
        return ModelCard(
            base_model_hf=self.base_model,
            model_name=self.model_name,
            major=major,
            minor=minor,
            patch=patch,
            model_description=model_description or self.description,
            training_data=list(self.training_batch_names),
            output_schema=self.schema,
        )

    def valid_model_card_version(self, model_card: ModelCard) -> bool:
        return (
            not len(
                AWS.list_s3_objects(
                    self.region,
                    self.bucket,
                    LoRATrainer._get_uploaded_model_folder_prefix(model_card),
                )
            )
            > 0
        )

    def _prepare_training_data(self, output_file: str) -> str:
        return TrainingDataHandler.prepare(
            schema=self.schema,
            system_prompt=self.prompt_builder.build_main_prompt(),
            training_batch_names=self.training_batch_names,
            base_model=self.base_model,
            max_seq_length=self.max_seq_length,
            bucket=self.bucket,
            s3_prefix="trainingdata",
            output_file=output_file,
            region=self.region,
            shuffle=True,
        )

    def _upload_model_folder(self, target_folder: str, model_card: ModelCard) -> None:
        target_path = Path(target_folder)
        # Write model_card.yaml into the folder (same bytes as the public tarball, .yaml filename).
        (target_path / "model_card.yaml").write_bytes(model_card.to_yaml_bytes())

        for item in target_path.iterdir():
            if not item.is_file():
                continue  # fused output is flat; skip any sub-dirs
            if not AWS.upload_file(
                region_name=self.region,
                file_name=str(item),
                bucket=self.bucket,
                object_name=item.name,
                path=LoRATrainer._get_uploaded_model_folder_prefix(model_card),
            ):
                raise ValueError(f"Failed to upload {item.name} to build bucket")

    def _archive_and_upload(
        self,
        target_folder: str,
        model_card: ModelCard,
        bucket: str = "aicentre-nlpteam-mesa-public",
    ) -> bool:
        target_path = Path(target_folder)
        archive_name = f"{model_card.model_identifier}.tar.gz"
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
        if not AWS.upload_file(
            region_name=self.region,
            file_name=str(archive_path),
            bucket=bucket,
            object_name=archive_name,
            path=self.model_name,
        ):
            raise ValueError("Failed to upload merged model weights")
        return True

    def _publish(
        self, target_folder: str, model_card: ModelCard, push_public: bool
    ) -> None:
        self._upload_model_folder(target_folder, model_card)
        if push_public:
            self._archive_and_upload(target_folder, model_card)
