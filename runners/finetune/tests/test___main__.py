from unittest.mock import MagicMock

import pytest

from finetune_runner.__main__ import FinetuneMLXRunner


@pytest.fixture
def runner() -> FinetuneMLXRunner:
    return FinetuneMLXRunner.model_validate(
        {"training_batch_name": "foo", "model_name": "bar", "version": "1.2.3"}
    )


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
