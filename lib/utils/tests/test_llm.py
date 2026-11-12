import os
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from utils.llm import LLM, Choice, ModelResponse, ResponseMessage


@dataclass
class CompletionMocks:
    bedrock: MagicMock
    completion: MagicMock


@dataclass
class BedrockMocks:
    client: MagicMock


@pytest.fixture
def completion_mocks(
    mocker: MockerFixture, model_response: ModelResponse
) -> CompletionMocks:
    mocks: CompletionMocks = CompletionMocks(
        mocker.patch("utils.llm.LLM.bedrock", return_value=model_response),
        mocker.patch("utils.llm.completion"),
    )
    choice: MagicMock = MagicMock()
    choice.message.content = "foo"
    choice.finish_reason = "stop"
    mocks.completion.return_value.choices = [choice]
    return mocks


@pytest.fixture
def bedrock_mocks(mocker: MockerFixture) -> BedrockMocks:
    mocker.patch.dict(os.environ)
    mocks: BedrockMocks = BedrockMocks(mocker.patch("utils.llm.boto3.client"))
    mocks.client.return_value.converse.return_value = {
        "output": {"message": {"content": [{"reasoningContent": {}}, {"text": "foo"}]}},
        "stopReason": "end_turn",
    }
    return mocks


class TestCompletion:
    def test_completion_bedrock_model_completes_through_bedrock(
        self, completion_mocks: CompletionMocks, model_response: ModelResponse
    ) -> None:
        assert LLM.completion("bedrock/foo", "bar", "baz", "qux") == model_response
        completion_mocks.bedrock.assert_called_once_with(
            "foo", "bar", "baz", "qux", 8192, 0.001
        )

    def test_completion_other_model_returns_first_choice(
        self, completion_mocks: CompletionMocks
    ) -> None:
        assert LLM.completion("foo", "bar", "baz", "qux") == ModelResponse(
            choices=[
                Choice(message=ResponseMessage(content="foo"), finish_reason="stop")
            ]
        )
        assert completion_mocks.completion.call_args.kwargs["model"] == "foo"


class TestBedrock:
    def test_bedrock_model_returns_text_from_converse(
        self, bedrock_mocks: BedrockMocks
    ) -> None:
        assert LLM.bedrock("foo.bar", "baz", "qux", "quux", region_name="corge") == (
            ModelResponse(
                choices=[
                    Choice(
                        message=ResponseMessage(content="foo"), finish_reason="end_turn"
                    )
                ]
            )
        )
        bedrock_mocks.client.assert_called_once_with(
            "bedrock-runtime", region_name="corge"
        )
        converse: MagicMock = bedrock_mocks.client.return_value.converse
        assert converse.call_args.kwargs["modelId"] == "foo.bar"
        assert converse.call_args.kwargs["system"] == [{"text": "baz"}]
        assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "quux"

    def test_bedrock_without_system_prompt_omits_system(
        self, bedrock_mocks: BedrockMocks
    ) -> None:
        LLM.bedrock("foo.bar", None, "qux", "quux")
        assert (
            "system" not in bedrock_mocks.client.return_value.converse.call_args.kwargs
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
