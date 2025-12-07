from re import Match, DOTALL, search
from typing import Any, Literal

from litellm import Usage, completion, ModelResponse
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
    @staticmethod
    def completion(
        model_name: str,
        system_prompt: str | None,
        user_prompt: str,
        api_key: str,
        max_tokens: int = 8192,
        temperature: float = 0.001,
        **kwargs: Any,
    ) -> ModelResponse | None:
        """Use an LLM for inference.

        Args:
            model_name (str): The name of the LLM
            system_prompt (str): The system prompt to use
            user_prompt (str): The user prompt to use
            api_key (str): API key to access the remote API
            max_tokens (int): Maximum output tokens. Defaults to 8192.
            temperature (float): Model randomness. Defaults to 0.001.

        Returns:
            ModelResponse: The model's prediction (LiteLLM wrapper object)

        """
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"content": system_prompt, "role": "system"})
        messages.append({"content": user_prompt, "role": "user"})
        return completion(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            api_key=api_key,
            stream=False,
            **kwargs,
        )

    @staticmethod
    def extract_output_content(response_text: str) -> tuple[bool, str, str]:
        """Extract the json portion of an LLM schema standardisation response.

        Args:
            response_text (str): The full response text

        Returns:
            tuple: Whether the extraction was successful, a status message,
                and the extracted (or full) content

        """
        pattern: str = r"<OUTPUT>(.*?)</OUTPUT>"
        match: Match[str] | None = search(pattern, response_text, DOTALL)
        if match:
            content: str = match.group(1).strip()
            return (
                True,
                f"Successfully extracted content from <OUTPUT> tags (length={len(content)} chars)",
                content,
            )
        else:
            return (
                False,
                "No <OUTPUT> tags found in response, using full response text",
                response_text.strip(),
            )
