import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from utils.aws import AWS

"""
llm_clients.py - LLM client abstractions for calling different API providers.
"""


class LLMClient(ABC):
    """
    Abstract base class for any clients
    """

    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 4000):
        """
        Initialise Gemini client.

        Args:
            model:
                Model name (e.g., 'gemini-2.5-flash')
            temperature:
                Sampling temperature
            max_tokens:
                Max tokens to generate
        """
        self._logger = logging.getLogger(__name__)
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY not found in environment variables")

        # Store generation parameters for API calls
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._logger.info(
            f"Initialized client with model={model}, temperature={temperature}, max_tokens={max_tokens}"
        )

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt:
                Prompt to send to the LLM

        Returns:
            Raw response text from the LLM
        """
        pass


class GeminiClient(LLMClient):
    """
    Client for Google Gemini API
    """

    def generate(self, prompt: str) -> str:
        """
        Generate response from Gemini.
        """

        self._logger.debug(f"Sending prompt to Gemini (length={len(prompt)} chars)")

        try:
            # Disable all safety filters to allow medical/technical content generation
            safety_settings = [
                {
                    "category": self.genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": self.genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": self.genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": self.genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
            ]

            response = AWS.bedrock_completion(
                "gemini/" + self.model_name,
                None,
                prompt,
                self.api_key,
                self.max_tokens,
                self.temperature,
                safety_settings=safety_settings,
            )

            # Check if response was blocked by safety filters
            if not response.choices[0].message.content:
                finish_reason = response.choices[0].finish_reason
                self._logger.error(
                    f"Gemini blocked response. Finish reason: {finish_reason}"
                )
                raise ValueError(
                    f"Response blocked by Gemini. Finish reason: {finish_reason}"
                )

            result = response.choices[0].message.content
            self._logger.debug(
                f"Received response from Gemini (length={len(result)} chars)"
            )
            return result

        except Exception as e:
            self._logger.error(f"Error generating from Gemini: {e}")
            raise


class AnthropicClient(LLMClient):
    """
    Client for Anthropic API
    """

    def generate(self, prompt: str) -> str:
        self._logger.debug(f"Sending prompt to Claude (length={len(prompt)} chars)")

        try:
            response = AWS.bedrock_completion(
                self.model_name,
                None,
                prompt,
                self.api_key,
                self.max_tokens,
                self.temperature,
            )

            result = response.choices[0].message.content
            self._logger.debug(
                f"Received response from Claude (length={len(result)} chars)"
            )
            return result

        except Exception as e:
            self._logger.error(f"Error generating from Claude: {e}")
            raise


class LocalClient(LLMClient):
    """
    Client for local OpenAI-compatible endpoint
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 4000,
    ):
        """
        Initialize local OpenAI-compatible client.

        Args:
            base_url:
                Base URL for the API (e.g., 'http://localhost:1234/v1')
            model:
                Model name
            temperature:
                Sampling temperature
            max_tokens:
                Max tokens to generate
        """

        super().__init__(model, temperature, max_tokens)
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        """
        Generate response from local API.
        """

        self.__logger.debug(f"Sending prompt to local API (length={len(prompt)} chars)")

        try:
            response = AWS.bedrock_completion(
                self.model_name,
                None,
                prompt,
                self.api_key,
                self.max_tokens,
                self.temperature,
                api_base=self.base_url,
            )

            result = response.choices[0].message.content
            self.__logger.debug(
                f"Received response from local API (length={len(result)} chars)"
            )
            return result

        except Exception as e:
            self.__logger.error(f"Error generating from local API: {e}")
            raise


def create_llm_client(llm_config: dict) -> Optional[LLMClient]:
    """
    Factory function to create the appropriate LLM client based on config.

    Args:
        llm_config:
            Dictionary containing LLM configuration from pipeline.yml

    Returns:
        LLMClient instance or None if disabled
    """
    if not llm_config.get("enabled", False):
        print("LLM generation disabled")
        return None

    provider = llm_config.get("provider", "none")

    # Return None if no provider configured
    if provider == "none":
        print("LLM provider set to 'none'")
        return None

    elif provider == "gemini":
        config = llm_config["gemini"]
        return GeminiClient(
            model=config["model"],
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_tokens", 4000),
        )

    elif provider == "anthropic":
        config = llm_config["anthropic"]
        return AnthropicClient(
            model=config["model"],
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_tokens", 4000),
        )

    elif provider == "local":
        config = llm_config["local"]
        # Read base_url and model from environment variables
        base_url = os.getenv("LOCAL_LLM_BASE_URL")
        model = os.getenv("LOCAL_LLM_MODEL")

        if not base_url:
            raise ValueError("LOCAL_LLM_BASE_URL not found in environment variables")
        if not model:
            raise ValueError("LOCAL_LLM_MODEL not found in environment variables")

        return LocalClient(
            base_url=base_url,
            model=model,
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_tokens", 4000),
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
