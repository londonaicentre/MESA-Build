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
