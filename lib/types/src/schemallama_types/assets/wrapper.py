import json
from abc import ABC, abstractmethod
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any

from pydantic import BaseModel


class SchemaLlamaAssets(ABC):
    def __init__(self, base_dir: str, schema: type[BaseModel]) -> None:
        self._base_dir: Traversable = files(base_dir)
        self._schema: type[BaseModel] = schema

    def _load(self, folder: str, file: str) -> str:
        return self._base_dir.joinpath(f"{folder}/{file}").read_text()

    def validate_json(self, json_str: str) -> BaseModel:
        """Validate a schema json string

        Args:
            json_str (str): The json string to validate

        Returns:
            The validated schema as a BaseModel instance

        """
        parsed: dict[str, Any] = json.loads(json_str)
        return self._schema(**parsed)

    def load_user_prompt_template(self, template_name: str) -> str:
        """Load a user prompt template from wrapped assets.
                Template is assumed to be stored in the form
                `userprompt_{template}.md`

        Args:
            template_name (str): The name of the user prompt template
                to load

        Returns:
            str: The text of the template as a string

        """
        return self._load("prompts", f"userprompt_{template_name}.md")

    @abstractmethod
    def load_system_prompt(self, file: str) -> str:
        pass

    @abstractmethod
    def load_bootstrap_user_prompt(self, instructions: str) -> str:
        pass

    @abstractmethod
    def load_datagen_user_prompt(self, row: dict[str, Any]) -> str:
        pass
