from unittest.mock import MagicMock, patch

from litellm import RateLimitError
import pytest

from utils.aws import AWS


@patch("utils.aws.boto3.client")
def test_upload_file_valid_input_succeeds(mock_client: MagicMock) -> None:
    mock_s3_client = MagicMock()
    mock_client.return_value = mock_s3_client
    AWS.upload_file("foo", "bar", "baz", "qux", "quux")
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


@patch("utils.llm.completion")
def test_completion_content_returned(
    mock_completion: MagicMock, model_response: MagicMock
) -> None:
    mock_completion.return_value = model_response
    AWS.bedrock_completion(
        "foo", "bar", "baz", "quux"
    ) == "The quick brown fox jumped over the lazy dog"
    AWS.bedrock_completion("foo", "bar", "baz", "quux") is not None


@patch("utils.llm.completion", side_effect=RateLimitError("", "", ""))
def test_completion_limit_raises_exception(
    mock_completion: MagicMock, model_response: MagicMock
) -> None:
    mock_completion.return_value = model_response
    with pytest.raises(RateLimitError):
        AWS.bedrock_completion("foo", "bar", "baz", "quux")


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
