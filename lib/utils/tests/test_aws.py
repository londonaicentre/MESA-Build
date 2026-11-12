import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
from litellm import RateLimitError
import pytest

from utils.aws import AWS
from utils.llm import ModelResponse


@patch("utils.aws.boto3.client")
def test_upload_file_valid_input_succeeds(mock_client: MagicMock) -> None:
    mock_s3_client = MagicMock()
    mock_client.return_value = mock_s3_client
    AWS.upload_file("foo", "bar", "baz", "qux", "quux", True)
    mock_s3_client.upload_file.assert_called_once_with("bar", "baz", "quux/qux")


@patch("utils.aws.boto3.client")
def test_download_file_valid_input_succeeds(mock_client: MagicMock) -> None:
    mock_s3_client = MagicMock()
    mock_client.return_value = mock_s3_client
    AWS.download_file("foo", "bar", "baz", "qux", "quux")
    mock_s3_client.download_file.assert_called_once_with("bar", "quux/qux", "baz")


@patch("utils.aws.boto3.client")
def test_download_file_with_wildcard_invalid_input_fails(
    mock_client: MagicMock,
) -> None:
    mock_object = {"Key": "foo"}
    mock_page = {"Contents": [mock_object]}
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [mock_page]
    mock_s3_client = MagicMock()
    mock_s3_client.get_paginator.return_value = mock_paginator
    mock_client.return_value = mock_s3_client
    assert not AWS.download_file_with_wildcard("foo", "bar", "baz", "qux", "quux/*")


@patch("utils.aws.LLM.completion")
def test_bedrock_completion_completes_in_eu_west_2(
    mock_completion: MagicMock, model_response: ModelResponse
) -> None:
    mock_completion.return_value = model_response
    assert AWS.bedrock_completion("foo", "bar", "baz", "quux") == model_response
    assert mock_completion.call_args.kwargs["aws_region_name"] == "eu-west-2"


@patch("utils.aws.time.sleep")
@patch("utils.aws.LLM.completion", side_effect=RateLimitError("", "", ""))
def test_bedrock_completion_rate_limited_raises_rate_limit_error_after_retries(
    mock_completion: MagicMock, mock_sleep: MagicMock
) -> None:
    with pytest.raises(RateLimitError):
        AWS.bedrock_completion("foo", "bar", "baz", "quux")
    assert mock_completion.call_count == 6
    assert mock_sleep.call_count == 5


def test_create_anthropic_bedrock_batch_entry_valid_fields_are_present() -> None:
    assert (
        AWS.create_anthropic_bedrock_batch_entry("", None, "")["modelInput"][
            "anthropic_version"
        ]
        == "bedrock-2023-05-31"
    )
    assert (
        AWS.create_anthropic_bedrock_batch_entry("foo", None, "")["recordId"] == "foo"
    )
    assert (
        AWS.create_anthropic_bedrock_batch_entry("", None, "bar")["modelInput"][
            "messages"
        ][0]["content"][0]["text"]
        == "bar"
    )
    assert (
        "system"
        not in AWS.create_anthropic_bedrock_batch_entry("", None, "bar")[
            "modelInput"
        ].keys()
    )


@patch("utils.aws.boto3.client")
def test_create_model_invocation_job_valid_input_succeeds(
    mock_client: MagicMock,
) -> None:
    mock_bedrock_client = MagicMock()
    mock_client.return_value = mock_bedrock_client
    AWS.create_model_invocation_job("foo", "bar", "baz", "qux", "quux", "foobar")
    mock_bedrock_client.create_model_invocation_job.assert_called_once()


@patch("utils.aws.boto3.client")
def test_create_model_invocation_job_local_path_reads_base_name(
    mock_client: MagicMock,
) -> None:
    mock_bedrock_client = MagicMock()
    mock_client.return_value = mock_bedrock_client
    AWS.create_model_invocation_job("foo", "bar", "baz/qux", "quux", "corge", "grault")
    assert mock_bedrock_client.create_model_invocation_job.call_args.kwargs[
        "inputDataConfig"
    ]["s3InputDataConfig"]["s3Uri"] == ("s3://quux/foo/input/qux")


@patch("utils.aws.AWS.create_model_invocation_job")
@patch("utils.aws.AWS.upload_file")
def test_run_batch_inference_local_path_uploads_under_base_name(
    mock_upload_file: MagicMock,
    mock_create_model_invocation_job: MagicMock,
) -> None:
    AWS.run_batch_inference("foo", "bar", "baz/qux", "quux", "corge", "grault")
    mock_upload_file.assert_called_once_with(
        "grault", "baz/qux", "quux", path="foo/input"
    )
    mock_create_model_invocation_job.assert_called_once_with(
        "foo", "bar", "baz/qux", "quux", "corge", "grault"
    )


@patch("utils.aws.boto3.client")
def test_create_model_invocation_job_rejected_raises_client_error(
    mock_client: MagicMock,
) -> None:
    mock_bedrock_client = MagicMock()
    mock_bedrock_client.create_model_invocation_job.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "bad"}},
        "CreateModelInvocationJob",
    )
    mock_client.return_value = mock_bedrock_client
    with pytest.raises(ClientError):
        AWS.create_model_invocation_job("foo", "bar", "baz", "qux", "quux", "foobar")


@patch("utils.aws.AWS.download_file_with_wildcard")
def test_download_batch_output_file_found_downloads_file(
    mock_download_file_with_wildcard: MagicMock,
) -> None:
    mock_download_file_with_wildcard.return_value = True
    AWS.download_batch_output("foo", "bar", "baz", "qux/quux")
    mock_download_file_with_wildcard.assert_called_once_with(
        "foo", "bar", "qux/quux", "quux", "baz/output/*"
    )


@patch("utils.aws.AWS.download_file_with_wildcard")
def test_download_batch_output_file_missing_raises_value_error(
    mock_download_file_with_wildcard: MagicMock,
) -> None:
    mock_download_file_with_wildcard.return_value = False
    with pytest.raises(ValueError, match="Error downloading file"):
        AWS.download_batch_output("foo", "bar", "baz", "qux")


def test_parse_batch_output_valid_file_returns_batch_outputs(tmp_path: Path) -> None:
    batch_output_file: Path = tmp_path / "batch.jsonl.out"
    batch_output_file.write_text(
        json.dumps(
            {
                "recordId": "0",
                "modelInput": {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 10,
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "foo"}]}
                    ],
                },
                "modelOutput": {
                    "model": "foo",
                    "id": "bar",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "bar"}],
                    "stop_reason": "end_turn",
                    "usage": {},
                },
            }
        )
        + "\n"
    )
    batch_outputs = AWS.parse_batch_output(str(batch_output_file))
    assert len(batch_outputs.outputs) == 1
    assert batch_outputs.outputs[0].recordId == "0"
    assert batch_outputs.outputs[0].modelOutput.content[0].text == "bar"


@patch("utils.aws.AWS.parse_batch_output")
@patch("utils.aws.AWS.download_batch_output")
def test_get_batch_inference_outputs_downloads_then_parses(
    mock_download_batch_output: MagicMock,
    mock_parse_batch_output: MagicMock,
) -> None:
    mock_parse_batch_output.return_value = "parsed"
    result = AWS.get_batch_inference_outputs("foo", "bar", "baz", "qux")
    mock_download_batch_output.assert_called_once_with("foo", "bar", "baz", "qux")
    mock_parse_batch_output.assert_called_once_with("qux")
    assert result == "parsed"
