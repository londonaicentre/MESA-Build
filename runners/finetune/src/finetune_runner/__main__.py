import logging
import os
from abc import abstractmethod
from enum import Enum
from importlib import import_module
from typing import Self

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, CliApp, CliSuppress, SettingsConfigDict

from finetune.mlx_trainer import MLXLoRATrainer
from utils.prompt import BasePromptBuilder


class Schema(str, Enum):
    onco = "onco"
    geno = "geno"


class FinetuneRunner(BaseSettings):
    model_config = SettingsConfigDict(cli_kebab_case=True)

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

    def _load_schema(self) -> tuple[type[BaseModel], BasePromptBuilder]:
        package, model_name = {
            Schema.onco: ("oncoschema", "OncologyModel"),
            Schema.geno: ("genoschema", "GenomicTestReport"),
        }[self.schema_name]
        return (
            getattr(import_module(f"{package}.schema"), model_name),
            import_module(f"{package}.prompt_builder").PromptBuilder(),
        )

    @abstractmethod
    def cli_cmd(self) -> None:
        """Run the fine-tuning job for the selected backend and publish its model card."""
        ...


class FinetuneMLXRunner(FinetuneRunner):
    model_config = SettingsConfigDict(cli_prog_name="mesa-build-mlx-finetune")

    def cli_cmd(self) -> None:
        logging.info("== mesa-build (finetune) ==")
        logging.info("Collecting configuration")
        logging.info(f"  Using config at {self.config}")

        schema, prompt_builder = self._load_schema()
        trainer: MLXLoRATrainer = MLXLoRATrainer(
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
        trainer.run()
        trainer.post_process(
            trainer.build_model_card(self.major, self.minor, self.patch),
            push_public=False,
        )
        logging.info(
            "Post-processing complete — merged model uploaded to the build bucket."
        )


def mlx() -> None:
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
