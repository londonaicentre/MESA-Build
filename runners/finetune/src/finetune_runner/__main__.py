import logging
import os
import signal
from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from importlib import import_module
from pathlib import Path
from types import FrameType
from typing import Self

from mesa_types.model_card import ModelCard
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, CliApp, CliSuppress, SettingsConfigDict

from finetune.mlx_trainer import MLXLoRATrainer
from finetune.trainer import LoRATrainer
from utils.prompt import BasePromptBuilder


def _cancel_to_interrupt(_signum: int, _frame: FrameType | None) -> None:
    # workflow cancellation -> ctrl + c
    raise KeyboardInterrupt


class Schema(str, Enum):
    onco = "onco"
    geno = "geno"


class FinetuneRunner(BaseSettings):
    model_config = SettingsConfigDict(cli_kebab_case=True, cli_implicit_flags=True)

    config: str = Field(
        "config.yaml",
        validation_alias=AliasChoices("config", "c"),
        description="Path to YAML config file",
    )
    training_batch_name: str = Field(
        validation_alias=AliasChoices("training_batch_name", "b"),
        description="Training batch to finetune on, found in S3: s3://aicentre-nlpteam-mesa-build/trainingdata/<batch_name>/",
    )
    model: str = Field(
        validation_alias=AliasChoices("model_name", "m"),
        description="Key identifier for the model family, that also becomes the top-level folder in S3 (models/<model_name>/...)",
    )
    schema_name: Schema = Field(
        Schema.onco,
        validation_alias=AliasChoices("schema", "s"),
        description="Extraction schema to finetune for",
    )
    description: str = Field(
        "",
        validation_alias=AliasChoices("description", "d"),
        description="Job description, used for naming and the working directory (defaults to the model name)",
    )
    version: str = Field(
        "1.0.0",
        validation_alias=AliasChoices("version", "v"),
        description="Semantic version (<major.minor.patch>) recorded on the model card",
    )
    train: bool = Field(
        False,
        description="Train only; write the serialised trainer spec to --spec-out",
    )
    post_process: bool = Field(False, description="Post-process only; requires --spec")
    spec_out: str = Field(
        "", description="File to write the serialised trainer spec to (with --train)"
    )
    spec: str = Field(
        "",
        description="Serialised trainer spec: inline JSON, or a path to the --spec-out file from a prior --train run",
    )
    major: CliSuppress[int] = 0
    minor: CliSuppress[int] = 0
    patch: CliSuppress[int] = 0

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        assert len(value.split(".")) == 3 and all(
            part.isdigit() for part in value.split(".")
        ), "version must be in <major.minor.patch> form"
        return value

    @model_validator(mode="after")
    def _default_description(self) -> Self:
        if not self.description:
            self.description = self.model
        return self

    @model_validator(mode="after")
    def _split_version(self) -> Self:
        self.major, self.minor, self.patch = (
            int(part) for part in self.version.split(".")
        )
        return self

    @model_validator(mode="after")
    def _validate_stage(self) -> Self:
        assert not (self.train and self.post_process), (
            "pass at most one of --train / --post-process"
        )
        assert self.spec or not self.post_process, "--post-process requires --spec"
        assert self.spec_out or not self.train, "--train requires --spec-out"
        return self

    def _load_schema(self) -> tuple[type[BaseModel], BasePromptBuilder]:
        package, model_name = {
            Schema.onco: ("oncoschema", "OncologyModel"),
            Schema.geno: ("genoschema", "GenomicTestReport"),
        }[self.schema_name]
        return (
            getattr(import_module(f"{package}.schema"), model_name),
            import_module(f"{package}.prompt_builder").PromptBuilder(),
        )

    def _build_validated_model_card(self, trainer: LoRATrainer) -> ModelCard:
        model_card: ModelCard = trainer.build_model_card(
            self.major, self.minor, self.patch
        )
        if not trainer.valid_model_card_version(model_card):
            raise ValueError(
                f"Existing model (card) with version: {model_card.model_version}. Please bump."
            )
        return model_card

    @abstractmethod
    def cli_cmd(self) -> None:
        """Run the fine-tuning job for the selected backend and publish its model card."""
        ...


class FinetuneMLXRunner(FinetuneRunner):
    model_config = SettingsConfigDict(cli_prog_name="mesa-build-mlx-finetune")

    def _make_trainer(self) -> MLXLoRATrainer:
        schema, prompt_builder = self._load_schema()
        return MLXLoRATrainer(
            schema=schema,
            prompt_builder=prompt_builder,
            training_batch_names=[self.training_batch_name],
            config_path=self.config,
            aws_config={
                "bucket": "aicentre-nlpteam-mesa-build",
                "region": "eu-west-2",
            },
            model_name=self.model,
            description=self.description,
            work_dir="data/models",
            quantize=None,
        )

    def _write_spec(self, trainer: MLXLoRATrainer) -> None:
        Path(self.spec_out).write_text(trainer.to_json())
        logging.info(f"Wrote trainer spec to {self.spec_out}")

    def _train(self, trainer: MLXLoRATrainer) -> None:
        logging.info(f"Starting training job: {trainer.job_id}")
        self._build_validated_model_card(trainer)
        config_path: str = trainer.setup()
        if self.spec_out:
            self._write_spec(trainer)
        trainer.train(config_path)
        logging.info(f"Job complete: {trainer.job_id}")

    def _train_and_post_process(self, trainer: MLXLoRATrainer) -> None:
        self._train(trainer)
        self._post_process(trainer)

    def _post_process(self, trainer: MLXLoRATrainer) -> None:
        trainer.post_process(
            trainer.build_model_card(self.major, self.minor, self.patch),
            push_public=False,
        )
        logging.info(
            "Post-processing complete - merged model uploaded to the build bucket."
        )

    def _guarded(
        self,
        trainer: MLXLoRATrainer,
        action: Callable[[MLXLoRATrainer], None],
        delete_on_success: bool,
    ) -> None:
        try:
            action(trainer)
        except KeyboardInterrupt:
            trainer.cleanup()
            raise
        else:
            if delete_on_success:
                trainer.cleanup()

    def cli_cmd(self) -> None:
        logging.info("== mesa-build (finetune) ==")
        logging.info("Collecting configuration")
        logging.info(f"  Using config at {self.config}")

        if self.post_process:
            self._guarded(MLXLoRATrainer.from_json(self.spec), self._post_process, True)
        elif self.train:
            self._guarded(self._make_trainer(), self._train, False)
        else:
            self._guarded(self._make_trainer(), self._train_and_post_process, True)


def mlx() -> None:
    signal.signal(signal.SIGTERM, _cancel_to_interrupt)
    log_level = getattr(
        logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=log_level,
    )
    CliApp.run(FinetuneMLXRunner)


if __name__ == "__main__":
    mlx()
