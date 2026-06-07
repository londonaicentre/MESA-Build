"""Shared fixtures for the finetune test suite.

Holds the cross-file stubs (prompt builder, a parseable config file, and the
trainer factory fixtures) so the HF and MLX trainer tests can share them.
"""

import textwrap
from pathlib import Path
from typing import Any, Protocol

import pytest

from finetune.hf_estimator import HuggingFaceLoRATrainer
from finetune.mlx_trainer import MLXLoRATrainer
from fixtures import SchemaFixture
from utils.prompt import BasePromptBuilder


class PromptBuilderFixture(BasePromptBuilder):
    """Minimal prompt builder whose main prompt is a known constant."""

    def __init__(self) -> None:
        pass

    def build_main_prompt(self) -> str:
        return "foo"


# A minimal neutral config that satisfies FinetuneConfig. base_model is "baz" so
# trainer assertions on self.base_model stay simple; the lora/mlx values are picked
# to make the translator maths in test_config easy to check (alpha/rank = 2.0).
CONFIG_YAML = textwrap.dedent(
    """\
    training:
      base_model: baz
      epochs: 2
      learning_rate: 0.0002
      batch_size: 4
      max_seq_length: 2048
      lora:
        rank: 8
        alpha: 16
        dropout: 0.05
        target_modules: [q_proj, k_proj]
    """
)


@pytest.fixture
def config_path(tmp_path: Path) -> str:
    """Write the minimal neutral config to a temp file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML)
    return str(path)


class HuggingFaceLoRATrainerFixture(HuggingFaceLoRATrainer):
    """Expose private path attributes for assertions."""

    def get_job_id(self) -> str:
        return self.job_id

    def get_s3_input_path(self) -> str:
        return self.s3_input_path

    def get_s3_output_path(self) -> str:
        return self.s3_output_path

    def get_s3_full_output_path(self) -> str:
        return self.s3_full_output_path


class TrainerFactory(Protocol):
    def __call__(self, **overrides: Any) -> HuggingFaceLoRATrainerFixture: ...


class MLXTrainerFactory(Protocol):
    def __call__(self, **overrides: Any) -> MLXLoRATrainer: ...


@pytest.fixture
def make_trainer(config_path: str) -> TrainerFactory:
    """Factory returning a HuggingFaceLoRATrainerFixture with sensible defaults.

    Override any constructor arg via keyword, e.g. ``make_trainer(description="x")``.
    Defaults to the shared parseable ``config_path``.
    """

    def _make(**overrides: Any) -> HuggingFaceLoRATrainerFixture:
        kwargs: dict[str, Any] = {
            "schema": SchemaFixture,
            "prompt_builder": PromptBuilderFixture(),
            "training_batch_names": ["bar"],
            "config_path": config_path,
            "aws_config": {"bucket": "qux", "region": "quux", "role": "corge"},
            "model_name": "foo",
            "description": "foo",
            "instance_type": "foo.bar1.2baz",
            "instance_count": 1,
            "transformers_version": "1.23",
            "pytorch_version": "1.2",
            "py_version": "foo123",
        }
        kwargs.update(overrides)
        return HuggingFaceLoRATrainerFixture(**kwargs)

    return _make


@pytest.fixture
def make_mlx_trainer(config_path: str) -> MLXTrainerFactory:
    """Factory returning an MLXLoRATrainer with sensible defaults.

    Override any constructor arg via keyword, e.g. ``make_mlx_trainer(quantize="q4")``.
    """

    def _make(**overrides: Any) -> MLXLoRATrainer:
        kwargs: dict[str, Any] = {
            "schema": SchemaFixture,
            "prompt_builder": PromptBuilderFixture(),
            "training_batch_names": ["bar"],
            "config_path": config_path,
            "aws_config": {"bucket": "qux", "region": "quux", "role": "corge"},
            "model_name": "foo",
            "description": "foo",
            "work_dir": "data/models",
        }
        kwargs.update(overrides)
        return MLXLoRATrainer(**kwargs)

    return _make
