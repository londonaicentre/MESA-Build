import os
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from google.genai.types import (
    FinishReason,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)
from pytest_mock import MockerFixture

from utils.llm import LLM, Choice, ModelResponse, ResponseMessage


@dataclass
class CompletionMocks:
    bedrock: MagicMock
    gemini: MagicMock
    openai: MagicMock


@dataclass
class BedrockMocks:
    client: MagicMock


@dataclass
class GeminiMocks:
    Client: MagicMock


@dataclass
class OpenAIMocks:
    OpenAI: MagicMock


@pytest.fixture
def completion_mocks(
    mocker: MockerFixture, model_response: ModelResponse
) -> CompletionMocks:
    mocks: CompletionMocks = CompletionMocks(
        mocker.patch("utils.llm.LLM.bedrock", return_value=model_response),
        mocker.patch("utils.llm.LLM.gemini", return_value=model_response),
        mocker.patch("utils.llm.LLM.openai", return_value=model_response),
    )
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


@pytest.fixture
def gemini_mocks(mocker: MockerFixture) -> GeminiMocks:
    mocks: GeminiMocks = GeminiMocks(mocker.patch("utils.llm.genai.Client"))
    response: MagicMock = mocks.Client.return_value.__enter__.return_value.models.generate_content.return_value
    response.text = "foo"
    response.candidates[0].finish_reason = FinishReason.STOP
    return mocks


@pytest.fixture
def openai_mocks(mocker: MockerFixture) -> OpenAIMocks:
    mocks: OpenAIMocks = OpenAIMocks(mocker.patch("utils.llm.OpenAI"))
    mocks.OpenAI.return_value.chat.completions.create.return_value.model_dump.return_value = {
        "id": "bar",
        "choices": [{"message": {"content": "foo"}, "finish_reason": "stop"}],
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

    def test_completion_gemini_model_completes_through_gemini(
        self, completion_mocks: CompletionMocks, model_response: ModelResponse
    ) -> None:
        assert (
            LLM.completion("gemini/foo", "bar", "baz", "qux", safety_settings=[])
            == model_response
        )
        completion_mocks.gemini.assert_called_once_with(
            "foo", "bar", "baz", "qux", 8192, 0.001, []
        )

    @pytest.mark.parametrize("model_name", ["foo", "openai/foo"])
    def test_completion_other_model_completes_through_openai(
        self,
        completion_mocks: CompletionMocks,
        model_response: ModelResponse,
        model_name: str,
    ) -> None:
        assert (
            LLM.completion(model_name, "bar", "baz", "qux", api_base="http://corge/v1")
            == model_response
        )
        completion_mocks.openai.assert_called_once_with(
            "foo", "bar", "baz", "qux", 8192, 0.001, "http://corge/v1"
        )


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


class TestGemini:
    def test_gemini_model_returns_text_from_gemini(
        self, gemini_mocks: GeminiMocks
    ) -> None:
        assert LLM.gemini("foo", "baz", "qux", "quux") == ModelResponse(
            choices=[
                Choice(message=ResponseMessage(content="foo"), finish_reason="STOP")
            ]
        )
        gemini_mocks.Client.assert_called_once_with(api_key="quux")
        generate: MagicMock = gemini_mocks.Client.return_value.__enter__.return_value.models.generate_content
        assert generate.call_args.kwargs["model"] == "foo"
        assert generate.call_args.kwargs["contents"] == "qux"
        assert generate.call_args.kwargs["config"].system_instruction == "baz"

    def test_gemini_safety_setting_dicts_pass_as_safety_settings(
        self, gemini_mocks: GeminiMocks
    ) -> None:
        LLM.gemini(
            "foo",
            "baz",
            "qux",
            "quux",
            safety_settings=[
                {
                    "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                }
            ],
        )
        generate: MagicMock = gemini_mocks.Client.return_value.__enter__.return_value.models.generate_content
        assert generate.call_args.kwargs["config"].safety_settings == [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            )
        ]

    def test_gemini_without_candidates_returns_no_finish_reason(
        self, gemini_mocks: GeminiMocks
    ) -> None:
        gemini_mocks.Client.return_value.__enter__.return_value.models.generate_content.return_value.candidates = None
        assert LLM.gemini("foo", "baz", "qux", "quux").choices[0].finish_reason is None


class TestOpenAI:
    @pytest.mark.parametrize(
        ("system_prompt", "messages"),
        [
            (None, [{"role": "user", "content": "qux"}]),
            (
                "baz",
                [
                    {"role": "system", "content": "baz"},
                    {"role": "user", "content": "qux"},
                ],
            ),
        ],
    )
    def test_openai_model_returns_response(
        self,
        openai_mocks: OpenAIMocks,
        system_prompt: str | None,
        messages: list[dict[str, str]],
    ) -> None:
        assert LLM.openai(
            "foo", system_prompt, "qux", "quux", api_base="http://corge/v1"
        ) == ModelResponse(
            choices=[
                Choice(message=ResponseMessage(content="foo"), finish_reason="stop")
            ]
        )
        openai_mocks.OpenAI.assert_called_once_with(
            api_key="quux", base_url="http://corge/v1"
        )
        create: MagicMock = openai_mocks.OpenAI.return_value.chat.completions.create
        assert create.call_args.kwargs["model"] == "foo"
        assert create.call_args.kwargs["messages"] == messages


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
