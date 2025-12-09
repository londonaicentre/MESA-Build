import json
from json import JSONDecodeError
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

from pydantic import BaseModel
from litellm import ModelResponse, Choices, Message
import pandas as pd
import pytest
from pytest import MonkeyPatch

from tests.types import TestSchema
from tests.claude import TestSampleGenerator


@pytest.fixture
def valid_model_response(valid_sample_json_output: str) -> ModelResponse:
    message: Message = Message(
        content="<OUTPUT>" + valid_sample_json_output + "</OUTPUT>"
    )
    choice: Choices = Choices(message=message)
    return ModelResponse(
        choices=[choice],
    )


def test_extract_json_from_response_valid_input_returns_dict(
    sample_generator: TestSampleGenerator,
) -> None:
    assert sample_generator.extract_json_from_response('{"foo":"bar"}') == {
        "foo": "bar"
    }
    assert sample_generator.extract_json_from_response(
        '<OUTPUT>{"foo":"bar"}</OUTPUT>'
    ) == {"foo": "bar"}
    assert sample_generator.extract_json_from_response('<FOO>{"foo":"bar"}</FOO>') == {
        "foo": "bar"
    }


def test_extract_json_from_response_invalid_input_raises_error(
    sample_generator: TestSampleGenerator,
) -> None:
    with pytest.raises(JSONDecodeError):
        sample_generator.extract_json_from_response("{foo}")


def test_validate_with_pydantic_valid_input_returns_true(
    sample_generator: TestSampleGenerator,
) -> None:
    valid: bool
    model: BaseModel | None
    valid, model = sample_generator.validate_with_pydantic(
        {"foo": {"foo": "bar"}, "baz": "qux", "qux": [1, 2, 3]}
    )
    assert valid
    assert isinstance(model, TestSchema)


def test_validate_with_pydantic_invalid_input_returns_false(
    sample_generator: TestSampleGenerator,
) -> None:
    valid: bool
    model: BaseModel | None
    valid, model = sample_generator.validate_with_pydantic(
        {"foo": {"foo": "bar"}, "baz": "qux", "qux": "foobar"}
    )
    assert not valid
    assert model is None


def test_extract_validate_and_save_sample_valid_input_writes_file(
    sample_generator: TestSampleGenerator,
    valid_sample_json_output: str,
    monkeypatch: MonkeyPatch,
) -> None:
    file_mock: MagicMock = mock_open()
    monkeypatch.setattr("builtins.open", file_mock)
    assert sample_generator.extract_validate_and_save_sample(
        "<OUTPUT>" + valid_sample_json_output + "</OUTPUT>",
        1,
    )
    file_mock.assert_called_once_with(
        f"samples_sonnet4/sample{1 + 1:04d}.json", "w", encoding="utf-8"
    )
    writes: list[Any] = [call.args[0] for call in file_mock().write.call_args_list]
    full_content: str = "".join(writes)

    assert json.loads(full_content) == {
        "content": "foo",
        "output": {"foo": {"foo": "bar"}, "baz": "qux", "qux": [1, 2, 3]},
    }
    assert json.loads(full_content) is not None


def test_extract_validate_and_save_sample_invalid_input_no_write(
    sample_generator: TestSampleGenerator, monkeypatch: MonkeyPatch
) -> None:
    file_mock: MagicMock = mock_open()
    monkeypatch.setattr("builtins.open", file_mock)
    assert not sample_generator.extract_validate_and_save_sample("foo", 1)
    assert not sample_generator.extract_validate_and_save_sample(
        '<OUTPUT>{"content": "foo"}</OUTPUT>',
        2,
    )
    assert not sample_generator.extract_validate_and_save_sample(
        '<OUTPUT>{"output": {"foo": {"foo": "bar"}, "baz": "qux", "qux": [1, 2, 3]}}</OUTPUT>',
        3,
    )
    file_mock.assert_not_called()


def test_extract_validate_and_save_sample_invalid_input_error_write(
    sample_generator: TestSampleGenerator, monkeypatch: MonkeyPatch
) -> None:
    file_mock: MagicMock = mock_open()
    monkeypatch.setattr("builtins.open", file_mock)
    assert not sample_generator.extract_validate_and_save_sample(
        '<OUTPUT>{"content": "foo", "output": {"foo": {"foo": "bar"}, "baz": "qux"}}</OUTPUT>',
        2,
    )
    file_mock.assert_called_once_with(
        f"samples_sonnet4/invalid_sample{2 + 1:04d}.json", "w", encoding="utf-8"
    )


@patch("datagen.claude.AWS")
def test_generate_sample_valid_content_returns_true(
    mock_aws: MagicMock,
    valid_model_response: ModelResponse,
    sample_generator: TestSampleGenerator,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_aws.bedrock_completion.return_value = valid_model_response
    file_mock: MagicMock = mock_open()
    monkeypatch.setattr("builtins.open", file_mock)
    assert sample_generator.generate_sample(
        pd.DataFrame([{}]),
        0,
    )


@patch("datagen.claude.AWS")
def test_generate_sample_invalid_content_returns_false(
    mock_aws: MagicMock, sample_generator: TestSampleGenerator
) -> None:
    message: Message = Message(content=None)
    choice: Choices = Choices(message=message)
    mock_aws.bedrock_completion.return_value = ModelResponse(
        choices=[choice],
    )
    assert not sample_generator.generate_sample(
        pd.DataFrame([{}]),
        0,
    )


@patch("datagen.claude.AWS")
def test_generate_sample_invalid_message_returns_false(
    mock_aws: MagicMock, sample_generator: TestSampleGenerator
) -> None:
    mock_aws.bedrock_completion.return_value = None
    assert not sample_generator.generate_sample(
        pd.DataFrame([{}]),
        0,
    )


@patch("os.makedirs")
@patch("os.listdir")
@patch("pandas.read_csv")
@patch("datagen.claude.AWS")
def test_process_bootstrap_rows_already_generated_returns_zero(
    mock_aws: MagicMock,
    mock_read_csv: MagicMock,
    mock_listdir: MagicMock,
    _: MagicMock,
    valid_model_response: ModelResponse,
    sample_generator: TestSampleGenerator,
) -> None:
    mock_read_csv.return_value = pd.DataFrame({"foo": ["bar", "baz"]})
    mock_listdir.return_value = ["foo", "bar"]
    mock_aws.bedrock_completion.return_value = valid_model_response
    assert sample_generator.process_bootstrap_rows(2) == (0, 0)
    assert not sample_generator.process_bootstrap_rows(2) == (2, 0)


@patch("os.makedirs")
@patch("os.listdir")
@patch("pandas.read_csv")
@patch("datagen.claude.AWS")
def test_process_bootstrap_rows_limited_bootstrap_returns_max(
    mock_aws: MagicMock,
    mock_read_csv: MagicMock,
    mock_listdir: MagicMock,
    _: MagicMock,
    monkeypatch: MonkeyPatch,
    valid_model_response: ModelResponse,
    sample_generator: TestSampleGenerator,
) -> None:
    mock_read_csv.return_value = pd.DataFrame({"foo": ["bar", "baz"]})
    mock_listdir.return_value = []
    mock_aws.bedrock_completion.return_value = valid_model_response
    file_mock: MagicMock = mock_open()
    monkeypatch.setattr("builtins.open", file_mock)
    assert sample_generator.process_bootstrap_rows(10) == (2, 0)
    assert not sample_generator.process_bootstrap_rows(10) == (3, 0)


@patch("os.makedirs")
@patch("os.listdir")
@patch("pandas.read_csv")
@patch("datagen.claude.AWS")
def test_process_bootstrap_rows_correct_samples_return_success(
    mock_aws: MagicMock,
    mock_read_csv: MagicMock,
    mock_listdir: MagicMock,
    _: MagicMock,
    monkeypatch: MonkeyPatch,
    valid_model_response: ModelResponse,
    sample_generator: TestSampleGenerator,
) -> None:
    mock_read_csv.return_value = pd.DataFrame({"foo": ["bar", "baz"]})
    mock_listdir.return_value = []
    mock_aws.bedrock_completion.return_value = valid_model_response
    file_mock: MagicMock = mock_open()
    monkeypatch.setattr("builtins.open", file_mock)
    assert sample_generator.process_bootstrap_rows(2) == (2, 0)
    assert file_mock.call_count == 2
    assert not sample_generator.process_bootstrap_rows(2) == (0, 0)
