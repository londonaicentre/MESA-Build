from unittest.mock import MagicMock, patch

from utils.llm import LLM


@patch("utils.llm.completion")
def test_completion_content_returned(
    mock_completion: MagicMock, model_response: MagicMock
) -> None:
    mock_completion.return_value = model_response
    LLM.completion(
        "foo", "bar", "baz", "quux"
    ) == "The quick brown fox jumped over the lazy dog"
    LLM.completion("foo", "bar", "baz", "quux") is not None


def test_extract_output_content_nested_data_returned() -> None:
    input: str = "<OUTPUT>foo</OUTPUT>"
    result: bool
    content: str
    result, _, content = LLM.extract_output_content(input)
    assert result and content == "foo"
    assert result and content != "bar"
    result, _, content = LLM.extract_output_content("bar")
    assert not result and content == "bar"
    assert not result and content != "foo"
