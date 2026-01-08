"""Base prompt builder for schemas"""

import json
from abc import ABC
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any

from pydantic import BaseModel


class BasePromptBuilder(ABC):
    """Base class for schema prompt builders

    (adapted from SchemaLlamaAssets wrapper)
    """

    def __init__(self, base_dir: str, schema: type[BaseModel]) -> None:
        """Initialize prompt builder.

        Args:
            base_dir: Package name (e.g. 'oncoschema', 'genoschema')
            schema: Pydantic model class for this schema
        """
        self._base_dir: Traversable = files(base_dir)
        self._schema: type[BaseModel] = schema

    def _load(self, folder: str, file: str) -> str:
        """Load a resource file from the package.

        Args:
            folder: Subdirectory name (e.g. 'examples')
            file: Filename (e.g. 'example.json')

        Returns:
            File contents as string
        """
        return self._base_dir.joinpath(f"{folder}/{file}").read_text()

    def _load_root(self, file: str) -> str:
        """Load a file from package root.

        Args:
            file: Filename (e.g. 'prompt_datagen.txt')

        Returns:
            File contents as string
        """
        return self._base_dir.joinpath(file).read_text()

    def validate_json(self, json_str: str) -> BaseModel:
        """Validate a schema json string.

        Args:
            json_str: The json string to validate

        Returns:
            The validated schema as a BaseModel instance
        """
        parsed: dict[str, Any] = json.loads(json_str)
        return self._schema(**parsed)

    def build_datagen_prompt(self) -> str:
        """Build data generation prompt with schema and example.

        Returns:
            Complete prompt with {SCHEMA} and {EXAMPLE} replaced
        """
        prompt = self._load_root("prompt_datagen.txt")
        schema_json = json.dumps(self._schema.model_json_schema(), indent=2)
        example_json = self._load("examples", "example.json")

        prompt = prompt.replace("{SCHEMA}", schema_json)
        prompt = prompt.replace("{EXAMPLE}", example_json)
        return prompt

    def build_main_prompt(self) -> str:
        """Build main/inference prompt with schema only.

        Returns:
            Complete prompt with {SCHEMA} replaced
        """
        prompt = self._load_root("prompt_main.txt")
        schema_json = json.dumps(self._schema.model_json_schema(), indent=2)

        prompt = prompt.replace("{SCHEMA}", schema_json)
        return prompt

    def validate_example(self) -> BaseModel:
        """Validate the canonical example against schema.

        Returns:
            Validated example output
        """
        from mesa_types.training_example import TrainingExample

        example_str = self._load("examples", "example.json")
        example_data = TrainingExample(**json.loads(example_str))

        # Validate the output field
        return self.validate_json(json.dumps(example_data.output))
