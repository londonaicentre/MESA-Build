from unittest.mock import MagicMock, patch

from pytest_mock import MockerFixture

from utils.llm import LLM, Choice, ModelResponse, ResponseMessage


@patch("utils.llm.completion")
def test_completion_response_returns_first_choice(mock_completion: MagicMock) -> None:
    choice: MagicMock = MagicMock()
    choice.message.content = "foo"
    choice.finish_reason = "stop"
    mock_completion.return_value.choices = [choice]
    assert LLM.completion("bar", "baz", "qux", "quux") == ModelResponse(
        choices=[Choice(message=ResponseMessage(content="foo"), finish_reason="stop")]
    )


def test_extract_output_content_nested_data_returned() -> None:
    input: str = "<output>foo</output>"
    result: bool
    content: str
    result, _, content = LLM.extract_output_content(input)
    assert result and content == "foo"
    assert result and content != "bar"
    result, _, content = LLM.extract_output_content("bar")
    assert not result and content == "bar"
    assert not result and content != "foo"


def test_claude_code_prompts_returns_stripped_output(mocker: MockerFixture) -> None:
    mock_run: MagicMock = mocker.patch("utils.llm.subprocess.run", autospec=True)
    mock_run.return_value.stdout = " <output>foo</output>\n"
    assert LLM.claude_code("bar", "baz", "qux") == "<output>foo</output>"
    arguments: list[str] = mock_run.call_args.args[0]
    assert arguments[:4] == ["claude", "-p", "--model", "bar"]
    assert arguments[arguments.index("--system-prompt") + 1] == "baz"
    assert mock_run.call_args.kwargs["input"] == "qux"
    assert mock_run.call_args.kwargs["check"]
    assert mock_run.call_args.kwargs["env"]["MAX_THINKING_TOKENS"] == "3000"
