import os
import subprocess
import tempfile
from re import Match, DOTALL, search
from typing import Any, Literal

import boto3
from google import genai
from google.genai.types import (
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    SafetySetting,
    SafetySettingOrDict,
)
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[TextContent]


class ModelInput(BaseModel):
    anthropic_version: str
    max_tokens: int
    messages: list[Message]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ResponseMessage(BaseModel):
    content: str | None


class Choice(BaseModel):
    message: ResponseMessage
    finish_reason: str | None


class ModelResponse(BaseModel):
    choices: list[Choice]


class ModelOutput(BaseModel):
    model: str
    id: str
    type: str
    role: str
    content: list[TextContent]
    stop_reason: str
    stop_sequence: str | None = None
    usage: Usage


class BatchOutput(BaseModel):
    modelInput: ModelInput
    modelOutput: ModelOutput
    recordId: str


class BatchOutputs(BaseModel):
    outputs: list[BatchOutput]


class LLM:
    __THINKING_TOKENS: int = 3000

    @staticmethod
    def completion(
        model_name: str,
        system_prompt: str | None,
        user_prompt: str,
        api_key: str,
        max_tokens: int = 8192,
        temperature: float = 0.001,
        safety_settings: list[SafetySettingOrDict] | None = None,
        api_base: str | None = None,
    ) -> ModelResponse:
        """Use an LLM for inference.

        Args:
            model_name (str): The name of the LLM
            system_prompt (str): The system prompt to use
            user_prompt (str): The user prompt to use
            api_key (str): API key to access the remote API
            max_tokens (int): Maximum output tokens. Defaults to 8192.
            temperature (float): Model randomness. Defaults to 0.001.
            safety_settings (list[SafetySettingOrDict], optional): Safety settings of a
                Gemini model. Defaults to Gemini's.
            api_base (str, optional): Base URL of an OpenAI-compatible API. Defaults
                to OpenAI's.

        Returns:
            ModelResponse: The model's response and why it finished

        """
        if model_name.startswith("bedrock/"):
            return LLM.bedrock(
                model_name.removeprefix("bedrock/"),
                system_prompt,
                user_prompt,
                api_key,
                max_tokens,
                temperature,
            )
        if model_name.startswith("gemini/"):
            return LLM.gemini(
                model_name.removeprefix("gemini/"),
                system_prompt,
                user_prompt,
                api_key,
                max_tokens,
                temperature,
                safety_settings,
            )
        return LLM.openai(
            model_name.removeprefix("openai/"),
            system_prompt,
            user_prompt,
            api_key,
            max_tokens,
            temperature,
            api_base,
        )

    @staticmethod
    def bedrock(
        model_name: str,
        system_prompt: str | None,
        user_prompt: str,
        api_key: str,
        max_tokens: int = 8192,
        temperature: float = 0.001,
        region_name: str | None = None,
    ) -> ModelResponse:
        """Use a Bedrock LLM for inference, through the Converse API.

        Args:
            model_name (str): The Bedrock model id
            system_prompt (str): The system prompt to use
            user_prompt (str): The user prompt to use
            api_key (str): Bedrock API key
            max_tokens (int): Maximum output tokens. Defaults to 8192.
            temperature (float): Model randomness. Defaults to 0.001.
            region_name (str, optional): AWS region of the model. Defaults to the
                configured region.

        Returns:
            ModelResponse: The model's response and why it finished

        """
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        response: dict[str, Any] = boto3.client(
            "bedrock-runtime", region_name=region_name
        ).converse(
            modelId=model_name,
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            **({} if system_prompt is None else {"system": [{"text": system_prompt}]}),
        )
        return ModelResponse(
            choices=[
                Choice(
                    message=ResponseMessage(
                        content=next(
                            (
                                block["text"]
                                for block in response["output"]["message"]["content"]
                                if "text" in block
                            ),
                            None,
                        )
                    ),
                    finish_reason=response["stopReason"],
                )
            ]
        )

    @staticmethod
    def gemini(
        model_name: str,
        system_prompt: str | None,
        user_prompt: str,
        api_key: str,
        max_tokens: int = 8192,
        temperature: float = 0.001,
        safety_settings: list[SafetySettingOrDict] | None = None,
    ) -> ModelResponse:
        """Use a Gemini LLM for inference, through the Gemini API.

        Args:
            model_name (str): The Gemini model name
            system_prompt (str): The system prompt to use
            user_prompt (str): The user prompt to use
            api_key (str): Gemini API key
            max_tokens (int): Maximum output tokens. Defaults to 8192.
            temperature (float): Model randomness. Defaults to 0.001.
            safety_settings (list[SafetySettingOrDict], optional): Safety settings for
                the request. Defaults to Gemini's.

        Returns:
            ModelResponse: The model's response and why it finished

        """
        with genai.Client(api_key=api_key) as client:
            response: GenerateContentResponse = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    safety_settings=None
                    if safety_settings is None
                    else [
                        SafetySetting.model_validate(setting)
                        for setting in safety_settings
                    ],
                ),
            )
        finish_reason: FinishReason | None = (
            response.candidates[0].finish_reason if response.candidates else None
        )
        return ModelResponse(
            choices=[
                Choice(
                    message=ResponseMessage(content=response.text),
                    finish_reason=None
                    if finish_reason is None
                    else finish_reason.value,
                )
            ]
        )

    @staticmethod
    def openai(
        model_name: str,
        system_prompt: str | None,
        user_prompt: str,
        api_key: str,
        max_tokens: int = 8192,
        temperature: float = 0.001,
        api_base: str | None = None,
    ) -> ModelResponse:
        """Use an LLM for inference, through an OpenAI-compatible API.

        Args:
            model_name (str): The model name, as the API serves it
            system_prompt (str): The system prompt to use
            user_prompt (str): The user prompt to use
            api_key (str): API key to access the API
            max_tokens (int): Maximum output tokens. Defaults to 8192.
            temperature (float): Model randomness. Defaults to 0.001.
            api_base (str, optional): Base URL of the API. Defaults to OpenAI's.

        Returns:
            ModelResponse: The model's response and why it finished

        """
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return ModelResponse.model_validate(
            OpenAI(api_key=api_key, base_url=api_base)
            .chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            .model_dump()
        )

    @staticmethod
    def claude_code(model_name: str, system_prompt: str, user_prompt: str) -> str:
        """Use Claude Code non-interactively, with no tools, settings or session.

        Args:
            model_name (str): The model, as accepted by `claude --model`
            system_prompt (str): The system prompt to use
            user_prompt (str): The user prompt to use

        Returns:
            str: The model's response, stripped of surrounding whitespace

        Raises:
            CalledProcessError: If Claude Code exits with an error

        """
        return subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                model_name,
                "--system-prompt",
                system_prompt,
                "--tools",
                "",
                "--strict-mcp-config",
                "--setting-sources",
                "",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--settings",
                '{"alwaysThinkingEnabled": true}',
            ],
            input=user_prompt,
            capture_output=True,
            text=True,
            check=True,
            cwd=tempfile.gettempdir(),
            env={
                **os.environ,
                "MAX_THINKING_TOKENS": str(LLM.__THINKING_TOKENS),
            },
        ).stdout.strip()

    @staticmethod
    def extract_output_content(response_text: str) -> tuple[bool, str, str]:
        """Extract the json portion of an LLM schema standardisation response.

        Args:
            response_text (str): The full response text

        Returns:
            tuple: Whether the extraction was successful, a status message,
                and the extracted (or full) content

        """
        pattern: str = r"<output>(.*?)</output>"
        match: Match[str] | None = search(pattern, response_text, DOTALL)
        if match:
            content: str = match.group(1).strip()
            return (
                True,
                f"Successfully extracted content from <output> tags (length={len(content)} chars)",
                content,
            )
        else:
            return (
                False,
                "No <output> tags found in response, using full response text",
                response_text.strip(),
            )
