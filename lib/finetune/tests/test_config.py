import math
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from finetune.config import FinetuneConfig

# A complete neutral config including the optional mlx block. lr_schedule omitted
# (defaults to None) so we can assert it is dropped from the mlx translation.
FULL_CONFIG_YAML = textwrap.dedent(
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
      mlx:
        iters: null
        num_layers: -1
        seed: 7
        save_every: 100
        steps_per_report: 10
        steps_per_eval: 200
        val_batches: 25
    """
)


@pytest.fixture
def full_config(tmp_path: Path) -> FinetuneConfig:
    path = tmp_path / "config.yaml"
    path.write_text(FULL_CONFIG_YAML)
    return FinetuneConfig.load(path)


class TestLoadConfig:
    # load_config parses a valid YAML into a populated model and rejects unknown / missing keys
    # (models use extra="forbid"). Real YAML written to tmp_path; nothing mocked.
    def test_load_config_returns_populated_config(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(FULL_CONFIG_YAML)
        cfg: FinetuneConfig = FinetuneConfig.load(path)
        assert cfg.training.base_model == "baz"
        assert cfg.training.epochs == 2
        assert cfg.training.lora.target_modules == ["q_proj", "k_proj"]
        assert cfg.training.mlx.seed == 7

    def test_load_config_unknown_key_raises(self, tmp_path: Path) -> None:
        # models use extra="forbid"; an unknown training key must be rejected.
        path = tmp_path / "config.yaml"
        path.write_text(FULL_CONFIG_YAML + "  unexpected_extra_key: 1\n")
        with pytest.raises(ValidationError):
            FinetuneConfig.load(path)

    def test_load_config_missing_required_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        # base_model omitted from training
        path.write_text(
            textwrap.dedent(
                """\
                training:
                  epochs: 2
                  learning_rate: 0.0002
                  batch_size: 4
                  max_seq_length: 2048
                  lora:
                    rank: 8
                    alpha: 16
                    dropout: 0.05
                    target_modules: [q_proj]
                """
            )
        )
        with pytest.raises(ValidationError):
            FinetuneConfig.load(path)


class TestLoadDefaultConfig:
    # The shipped default config parses and validates into a usable model.
    def test_load_default_config_validates(self) -> None:
        cfg: FinetuneConfig = FinetuneConfig.load_default()
        assert cfg.training.base_model
        assert cfg.training.lora.rank > 0


class TestToHfHyperparameters:
    # Translator: maps/renames config fields to SageMaker hyperparameter keys and comma-joins
    # target_modules.
    def test_maps_fields_and_renames(self, full_config: FinetuneConfig) -> None:
        assert full_config.to_hf_hyperparameters() == {
            "base_model": "baz",
            "num_epochs": 2,
            "learning_rate": 0.0002,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": "q_proj,k_proj",
            "per_device_train_batch_size": 4,
            "max_seq_length": 2048,
        }

    def test_target_modules_comma_joined(self, full_config: FinetuneConfig) -> None:
        assert (
            full_config.to_hf_hyperparameters()["lora_target_modules"]
            == "q_proj,k_proj"
        )


class TestToMlxConfig:
    # Translator: iters derived from sample count (or overridden), prefixed/scaled LoRA keys,
    # constant + passthrough fields, and the optional lr_schedule.
    def test_iters_derived_from_samples(self, full_config: FinetuneConfig) -> None:
        # ceil(10 / 4) * 2 = 6
        out = full_config.to_mlx_config(10)
        assert out["iters"] == math.ceil(10 / 4) * 2 == 6

    def test_iters_override_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(FULL_CONFIG_YAML.replace("iters: null", "iters: 123"))
        out = FinetuneConfig.load(path).to_mlx_config(10)
        assert out["iters"] == 123

    def test_lora_keys_prefixed_and_scale(self, full_config: FinetuneConfig) -> None:
        out = full_config.to_mlx_config(10)
        assert out["lora_parameters"]["keys"] == [
            "self_attn.q_proj",
            "self_attn.k_proj",
        ]
        assert out["lora_parameters"]["scale"] == 16 / 8
        assert out["lora_parameters"]["rank"] == 8

    def test_constants_and_passthrough(self, full_config: FinetuneConfig) -> None:
        out = full_config.to_mlx_config(10)
        assert out["model"] == "baz"
        assert out["train"] is True
        assert out["fine_tune_type"] == "lora"
        assert out["optimizer"] == "adamw"
        assert out["seed"] == 7

    def test_lr_schedule_omitted_when_none(self, full_config: FinetuneConfig) -> None:
        assert "lr_schedule" not in full_config.to_mlx_config(10)

    def test_lr_schedule_included_when_set(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            FULL_CONFIG_YAML.replace(
                "    val_batches: 25\n",
                "    val_batches: 25\n    lr_schedule:\n      name: cosine_decay\n",
            )
        )
        out = FinetuneConfig.load(path).to_mlx_config(10)
        assert out["lr_schedule"] == {"name": "cosine_decay"}
