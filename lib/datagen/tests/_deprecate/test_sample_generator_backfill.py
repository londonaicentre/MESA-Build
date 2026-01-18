from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
from pytest import MonkeyPatch
from litellm import ModelResponse

from tests.sample_generator_helper import TestSampleGenerator


@patch("os.listdir")
def test_find_missing_idx_on_missing_returns_index(
    mock_listdir: MagicMock, sample_generator: TestSampleGenerator
) -> None:
    mock_listdir.return_value = [
        "sample0001.json",
        "sample0002.json",
        "sample0004.json",
    ]
    assert sample_generator.find_missing_idx(4) == [2]


@patch("os.listdir")
def test_find_missing_idx_on_complete_returns_empty(
    mock_listdir: MagicMock, sample_generator: TestSampleGenerator
) -> None:
    mock_listdir.return_value = [
        "sample0001.json",
        "sample0002.json",
        "sample0003.json",
    ]
    assert sample_generator.find_missing_idx(3) == []


@patch("pandas.read_csv")
@patch("datagen.sample_generator.AWS")
def test_backfill_on_missing_outputs_sample(
    mock_aws: MagicMock,
    mock_read_csv: MagicMock,
    valid_model_response: ModelResponse,
    sample_generator: TestSampleGenerator,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_read_csv.return_value = pd.DataFrame({"foo": ["bar", "baz"]})
    mock_aws.bedrock_completion.return_value = valid_model_response
    file_mock: MagicMock = mock_open()
    monkeypatch.setattr("builtins.open", file_mock)
    assert sample_generator.backfill([0]) == (1, 0)
    file_mock.assert_called_once()


@patch("pandas.read_csv")
@patch("datagen.sample_generator.AWS")
def test_backfill_on_complete_outputs_unchanged(
    mock_aws: MagicMock,
    mock_read_csv: MagicMock,
    valid_model_response: ModelResponse,
    sample_generator: TestSampleGenerator,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_read_csv.return_value = pd.DataFrame({"foo": ["bar", "baz"]})
    mock_aws.bedrock_completion.return_value = valid_model_response
    file_mock: MagicMock = mock_open()
    monkeypatch.setattr("builtins.open", file_mock)
    assert sample_generator.backfill([]) == (0, 0)
