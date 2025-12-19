import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
from pytest import MonkeyPatch
import pytest

from tests.claude import TestSampleGenerator


def get_batch_input(sample_generation_input: str = "bar") -> str:
    return json.dumps(
        {
            "recordId": "0",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "system": "foo",
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": sample_generation_input}],
                    }
                ],
            },
        }
    )


@patch("pandas.read_csv")
def test_generate_batch_on_input_writes_file(
    mock_read_csv: MagicMock,
    sample_generator: TestSampleGenerator,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_read_csv.return_value = pd.DataFrame({"foo": ["bar", "baz"]})
    file_mock: MagicMock = mock_open()
    monkeypatch.setattr("builtins.open", file_mock)
    sample_generator.generate_batch(2)
    assert len(file_mock().write.call_args_list) == 4
    assert file_mock().write.call_args_list[0].args[0] == get_batch_input()


@patch("pandas.read_csv")
@patch("utils.aws.boto3.client")
def test_generate_via_batch_on_call_writes_job(
    _: MagicMock,
    mock_read_csv: MagicMock,
    sample_generator: TestSampleGenerator,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_read_csv.return_value = pd.DataFrame({"foo": ["bar", "baz"]})
    file_mock: MagicMock = mock_open()
    monkeypatch.setattr("builtins.open", file_mock)
    job_id: str = sample_generator.generate_via_batch(2, "foo", "bar")
    assert len(file_mock().write.call_args_list) == 5
    assert file_mock().write.call_args_list[4].args[0] == json.dumps({"job_id": job_id})


def get_batch_output(sample_json_output: str) -> str:
    return json.dumps(
        {
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "system": "",
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": ""}],
                    }
                ],
            },
            "modelOutput": {
                "id": "msg_bdrk",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": f"{sample_json_output}"}],
                "stop_reason": "end_turn",
                "stop_sequence": "null",
                "usage": {
                    "input_tokens": 11578,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 1138,
                },
            },
            "recordId": "40",
        }
    )


@patch("builtins.open")
def test_extract_batch_output_valid_output_extracts_to_file(
    mock_file: MagicMock,
    sample_generator: TestSampleGenerator,
    valid_sample_json_output: str,
) -> None:
    mock_file.return_value = mock_open(
        read_data=get_batch_output(valid_sample_json_output)
    )()
    print(get_batch_output(valid_sample_json_output))
    assert sample_generator.extract_batch_output() == (1, 0)


@patch("builtins.open")
def test_extract_batch_output_invalid_output_returns_error(
    mock_file: MagicMock, sample_generator: TestSampleGenerator
) -> None:
    mock_file.return_value = mock_open(read_data=get_batch_output("{}"))()
    assert sample_generator.extract_batch_output() == (0, 1)


@patch("builtins.open")
@patch("utils.aws.boto3.client")
def test_extract_batch_job_file_present_read_successfully(
    _: MagicMock,
    mock_file: MagicMock,
    sample_generator: TestSampleGenerator,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "exists", MagicMock(return_value=True))
    file_mock: MagicMock = mock_open(read_data=json.dumps({"job_id": "foo"}))()
    mock_file.return_value = file_mock
    with pytest.raises(ValueError):
        sample_generator.extract_batch_output("bar")
    assert len(file_mock.read.call_args_list) == 1
