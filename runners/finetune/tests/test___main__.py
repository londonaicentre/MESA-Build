from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from finetune.mlx_trainer import MLXLoRATrainer
from pydantic import ValidationError
from pytest_mock import MockerFixture

from finetune_runner.__main__ import FinetuneMLXRunner


def _runner(**overrides: Any) -> FinetuneMLXRunner:
    return FinetuneMLXRunner.model_validate(
        {
            "training_batch_name": "foo",
            "model_name": "bar",
            "version": "1.2.3",
            **overrides,
        }
    )


@pytest.fixture
def runner() -> FinetuneMLXRunner:
    return _runner()


class TestBuildValidatedModelCard:
    def test_valid_version_returns_model_card(self, runner: FinetuneMLXRunner) -> None:
        mock_trainer: MagicMock = MagicMock()
        mock_trainer.valid_model_card_version.return_value = True
        assert (
            runner._build_validated_model_card(mock_trainer)
            is mock_trainer.build_model_card.return_value
        )

    def test_invalid_version_raises(self, runner: FinetuneMLXRunner) -> None:
        mock_trainer: MagicMock = MagicMock()
        mock_trainer.valid_model_card_version.return_value = False
        with pytest.raises(ValueError):
            runner._build_validated_model_card(mock_trainer)

    def test_builds_model_card_with_version_parts(
        self, runner: FinetuneMLXRunner
    ) -> None:
        mock_trainer: MagicMock = MagicMock()
        mock_trainer.valid_model_card_version.return_value = True
        runner._build_validated_model_card(mock_trainer)
        mock_trainer.build_model_card.assert_called_once_with(1, 2, 3)


class TestValidateStage:
    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"train": True, "post_process": True}, "at most one"),
            ({"post_process": True}, "requires --spec"),
            ({"train": True}, "requires --spec-out"),
        ],
    )
    def test_invalid_stage_raises(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            _runner(**overrides)


class TestTrain:
    def test_validates_then_runs_and_returns_trainer(
        self, runner: FinetuneMLXRunner, mocker: MockerFixture
    ) -> None:
        trainer: MagicMock = mocker.patch.object(
            FinetuneMLXRunner, "_make_trainer"
        ).return_value
        validate: MagicMock = mocker.patch.object(
            FinetuneMLXRunner, "_build_validated_model_card"
        )
        assert runner._train() is trainer
        validate.assert_called_once_with(trainer)
        trainer.run.assert_called_once()


class TestPostProcess:
    def test_builds_card_and_post_processes(self, runner: FinetuneMLXRunner) -> None:
        trainer: MagicMock = MagicMock()
        runner._post_process(trainer)
        trainer.build_model_card.assert_called_once_with(1, 2, 3)
        trainer.post_process.assert_called_once_with(
            trainer.build_model_card.return_value, push_public=False
        )


class TestCliCmd:
    def test_default_trains_then_post_processes(
        self, runner: FinetuneMLXRunner, mocker: MockerFixture
    ) -> None:
        train: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_train")
        post: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_post_process")
        runner.cli_cmd()
        post.assert_called_once_with(train.return_value)

    def test_train_writes_spec_to_file(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        spec_file: Path = tmp_path / "spec.json"
        train: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_train")
        train.return_value.to_json.return_value = '{"foo": 1}'
        post: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_post_process")
        _runner(train=True, spec_out=str(spec_file)).cli_cmd()
        assert spec_file.read_text() == '{"foo": 1}'
        post.assert_not_called()

    def test_post_process_rebuilds_from_spec(self, mocker: MockerFixture) -> None:
        from_json: MagicMock = mocker.patch.object(MLXLoRATrainer, "from_json")
        train: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_train")
        post: MagicMock = mocker.patch.object(FinetuneMLXRunner, "_post_process")
        _runner(post_process=True, spec='{"foo": 1}').cli_cmd()
        from_json.assert_called_once_with('{"foo": 1}')
        post.assert_called_once_with(from_json.return_value)
        train.assert_not_called()
