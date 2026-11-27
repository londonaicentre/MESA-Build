import io
import json
import logging
import re
import tarfile
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import IO, Any, Final, Self

from mesa_types.model_card import ModelCard
from pydantic import BaseModel

from finetune.config import FinetuneConfig
from finetune.trainingdata_handler import TrainingDataHandler
from utils.aws import AWS
from utils.prompt import BasePromptBuilder
from utils.schema_resolver import SchemaResolver

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

    __LICENSE: Final[Path] = Path(__file__).parents[2] / "LICENSE.md"
    __MODEL_CARDS: Final[tuple[Path, Path]] = (
        Path("model_card.yml"),
        Path("model_card.yaml"),
    )
    __REPLACED: Final[tuple[Path, Path, Path]] = (Path("LICENSE.md"), *__MODEL_CARDS)

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

        self.job_id = LoRATrainer._make_job_id(description)

        # pass from an aws config dict (role unused for local training)
        self.bucket = aws_config["bucket"]
        self.region = aws_config["region"]
        self.role = aws_config.get("role", "")

    @staticmethod
    def _make_job_id(description: str) -> str:
        # job ID (sagemaker does not like underscores!)
        slug = re.sub(r"[^a-z0-9]+", "-", description.lower()).strip("-")[:40]
        return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug}".strip("-")

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
        SchemaResolver.install_schema_package(
            f"londonaicentre-{data['schema']['module'].split('.')[0]}", "", True
        )
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
            prompt_parameters=[
                parameters
                for batch_name in self.training_batch_names
                if (
                    parameters := TrainingDataHandler.load_prompt_parameters(
                        batch_name, self.bucket, "trainingdata", self.region
                    )
                )
                is not None
            ]
            or None,
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
            prompt_builder=self.prompt_builder,
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
        (target_path / "model_card.yml").write_bytes(model_card.to_yaml_bytes())

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

    @staticmethod
    def _add_model_card_and_license(
        tar: tarfile.TarFile, model_card_bytes: bytes
    ) -> None:
        tarinfo: tarfile.TarInfo = tarfile.TarInfo(name="model_card.yml")
        tarinfo.size = len(model_card_bytes)
        tar.addfile(tarinfo, io.BytesIO(model_card_bytes))
        tar.add(LoRATrainer.__LICENSE, arcname="LICENSE.md")

    @staticmethod
    def _upload_archive(
        region: str, archive_path: Path, bucket: str, model_name: str, force: bool
    ) -> None:
        if not AWS.upload_file(
            region_name=region,
            file_name=str(archive_path),
            bucket=bucket,
            object_name=archive_path.name,
            path=model_name,
            force=force,
        ):
            raise ValueError("Failed to upload merged model weights")

    @staticmethod
    def _archive_and_upload(
        region: str,
        target_path: Path,
        model_name: str,
        model_identifier: str,
        model_card_bytes: bytes,
        bucket: str,
        force: bool,
    ) -> None:
        archive_path: Path = target_path.parent / f"{model_identifier}.tar.gz"
        if force or not archive_path.exists():
            with tarfile.open(archive_path, "w:gz") as tar:
                for item in target_path.iterdir():
                    if Path(item.name) not in LoRATrainer.__MODEL_CARDS:
                        tar.add(item, arcname=item.name)
                LoRATrainer._add_model_card_and_license(tar, model_card_bytes)
        LoRATrainer._upload_archive(region, archive_path, bucket, model_name, force)

    def _publish(
        self, target_folder: str, model_card: ModelCard, push_public: bool
    ) -> None:
        self._upload_model_folder(target_folder, model_card)
        if push_public:
            LoRATrainer._archive_and_upload(
                self.region,
                Path(target_folder),
                self.model_name,
                model_card.model_identifier,
                model_card.to_yaml_bytes(),
                "aicentre-nlpteam-mesa-public",
                False,
            )

    @staticmethod
    def _download(region: str, bucket: str, file_path: Path, prefix: str) -> Path:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not AWS.download_file(
            region, bucket, str(file_path), file_path.name, prefix
        ):
            raise ValueError(f"Failed to download {prefix}/{file_path.name}")
        return file_path

    @staticmethod
    def make_public(
        model_name: str,
        model_identifier: str,
        work_dir: str,
        region: str = "eu-west-2",
        bucket: str = "aicentre-nlpteam-mesa-build",
        public_bucket: str = "aicentre-nlpteam-mesa-public",
    ) -> None:
        """Make a model in the build bucket public as a tarball in the public bucket.

        Args:
            model_name (str): Model name, the model's folder in both buckets.
            model_identifier (str): Model identifier (e.g. ``foo_1_2_3``), the
                model's folder under its name in the build bucket.
            work_dir (str): Local folder to download the model and write the
                tarball to.
            region (str): AWS region of both buckets. Defaults to
                ``"eu-west-2"``.
            bucket (str): Build bucket holding the model. Defaults to
                ``"aicentre-nlpteam-mesa-build"``.
            public_bucket (str): Bucket to publish the tarball to, overwriting
                any tarball already there. Defaults to
                ``"aicentre-nlpteam-mesa-public"``.

        Raises:
            ValueError: If a model file fails to download, the model has no
                model card, or the tarball fails to upload.

        """
        target_path: Path = Path(work_dir) / model_identifier
        prefix: str = f"models/{model_name}/{model_identifier}"
        for object in AWS.list_s3_objects(region, bucket, prefix + "/"):
            if not object["Key"].endswith("/"):
                LoRATrainer._download(
                    region, bucket, target_path / Path(object["Key"]).name, prefix
                )
        model_card_bytes: bytes | None = next(
            (
                (target_path / path).read_bytes()
                for path in LoRATrainer.__MODEL_CARDS
                if (target_path / path).exists()
            ),
            None,
        )
        if model_card_bytes is None:
            raise ValueError(f"{prefix} has no model card")
        LoRATrainer._archive_and_upload(
            region,
            target_path,
            model_name,
            model_identifier,
            model_card_bytes,
            public_bucket,
            True,
        )

    @staticmethod
    def repackage_model(
        model_name: str,
        model_identifier: str,
        work_dir: str,
        model_card: ModelCard | None = None,
        region: str = "eu-west-2",
        public_bucket: str = "aicentre-nlpteam-mesa-public",
    ) -> None:
        """Repackage a public model tarball with a model card and the current licence.

        Args:
            model_name (str): Model name, the tarball's folder in the public
                bucket.
            model_identifier (str): Model identifier (e.g. ``foo_1_2_3``), the
                tarball's name without ``.tar.gz``.
            work_dir (str): Local folder to download and rewrite the tarball in.
            model_card (ModelCard | None): Model card to write as
                ``model_card.yml``. Defaults to ``None``, which keeps the
                tarball's ``model_card.yml``, or failing that its
                ``model_card.yaml``, renamed to ``model_card.yml``.
            region (str): AWS region of the public bucket. Defaults to
                ``"eu-west-2"``.
            public_bucket (str): Bucket holding the tarball, which is
                overwritten. Defaults to ``"aicentre-nlpteam-mesa-public"``.

        Raises:
            ValueError: If the tarball fails to download or upload, or no model
                card is given and the tarball holds none.

        """
        archive_path: Path = Path(work_dir) / f"{model_identifier}.tar.gz"
        model_cards: dict[Path, bytes] = {}
        with (
            tarfile.open(
                LoRATrainer._download(
                    region,
                    public_bucket,
                    Path(work_dir) / "source" / archive_path.name,
                    model_name,
                ),
                "r:gz",
            ) as source,
            tarfile.open(archive_path, "w:gz") as tar,
        ):
            for member in source:
                path: Path = Path(member.name)
                content: IO[bytes] | None = source.extractfile(member)
                if path in LoRATrainer.__MODEL_CARDS and content is not None:
                    model_cards[path] = content.read()
                    content = io.BytesIO(model_cards[path])
                if path not in LoRATrainer.__REPLACED:
                    tar.addfile(member, content)
            model_card_bytes: bytes | None = (
                model_card.to_yaml_bytes()
                if model_card is not None
                else next(
                    filter(None, map(model_cards.get, LoRATrainer.__MODEL_CARDS)), None
                )
            )
            if model_card_bytes is None:
                raise ValueError(f"{model_name}/{archive_path.name} has no model card")
            LoRATrainer._add_model_card_and_license(tar, model_card_bytes)
        LoRATrainer._upload_archive(
            region, archive_path, public_bucket, model_name, True
        )
