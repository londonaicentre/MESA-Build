from typing import Any

from litellm import completion, ModelResponse


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
