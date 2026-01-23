"""Base prompt builder for schemas"""

import inspect
from abc import ABC
from importlib.resources import files
from importlib.resources.abc import Traversable

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

    def build_datagen_prompt(self) -> str:
        """Build data generation prompt with schema and example.

        Returns:
            Complete prompt with {SCHEMA} and {EXAMPLE} replaced
        """
        prompt = self._load_root("prompt_datagen.txt")

        # inserts full schema
        schema_module = inspect.getmodule(self._schema)
        if(schema_module is not None):
            schema_source = inspect.getsource(schema_module)
        else:
            raise ValueError('module not found')

        example_json = self._load("examples", "example.json")

        prompt = prompt.replace("{SCHEMA}", schema_source)
        prompt = prompt.replace("{EXAMPLE}", example_json)
        return prompt

    def build_main_prompt(self) -> str:
        """Build main/inference prompt with schema only.

        Returns:
            Complete prompt with {SCHEMA} replaced
        """
        prompt = self._load_root("prompt_main.txt")

        schema_module = inspect.getmodule(self._schema)
        if(schema_module is not None):
            schema_source = inspect.getsource(schema_module)
        else:
            raise ValueError('module not found')

        prompt = prompt.replace("{SCHEMA}", schema_source)
        return prompt
