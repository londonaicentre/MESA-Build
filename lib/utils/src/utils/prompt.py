"""Base prompt builder for schemas"""

import inspect
from abc import ABC
from functools import reduce
from importlib.resources import files
from importlib.resources.abc import Traversable

from pydantic import BaseModel


class BasePromptBuilder(ABC):
    """Base class for schema prompt builders

    (adapted from SchemaLlamaAssets wrapper)
    """

    def __init__(
        self,
        base_dir: str,
        schema: type[BaseModel],
        parameters: type[BaseModel] | None = None,
    ) -> None:
        """Initialize prompt builder.

        Args:
            base_dir: Package name (e.g. 'oncoschema', 'genoschema')
            schema: Pydantic model class for this schema
            parameters (type[BaseModel] | None): Model whose string fields name the
                caller-supplied prompt placeholders (e.g. `condition` fills
                `{CONDITION}`). Defaults to None, for a schema without parameters.
        """
        self._base_dir: Traversable = files(base_dir)
        self._schema: type[BaseModel] = schema
        self.__parameters: type[BaseModel] | None = parameters

    def get_parameters(self) -> type[BaseModel] | None:
        """Get the model of caller-supplied prompt parameters.

        Returns:
            type[BaseModel] | None: The parameters model, or None if the schema takes
                no parameters.

        """
        return self.__parameters

    def _substitute_parameters(
        self, prompt: str, parameters: dict[str, str] | None
    ) -> str:
        if self.__parameters is None:
            if parameters:
                raise ValueError("This schema does not take prompt parameters")
            return prompt
        if extra_keys := (parameters or {}).keys() - self.__parameters.model_fields:
            raise ValueError(f"Unknown prompt parameters: {sorted(extra_keys)}")
        return reduce(
            lambda text, item: text.replace("{" + item[0].upper() + "}", item[1]),
            [
                (name, value)
                for name, value in self.__parameters.model_validate(parameters or {})
                .model_dump()
                .items()
                if value is not None
            ],
            prompt,
        )

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

    def build_datagen_prompt(self, parameters: dict[str, str] | None = None) -> str:
        """Build data generation prompt with schema and example.

        Args:
            parameters (dict[str, str] | None): Values for the schema's prompt
                parameters, where an `example` value replaces the packaged example.
                Defaults to None.

        Returns:
            Complete prompt with parameters, {SCHEMA} and {EXAMPLE} replaced

        Raises:
            ValueError: If the parameters are missing, unknown or not accepted by
                this schema.
        """
        prompt = self._substitute_parameters(
            self._load_root("prompt_datagen.txt"), parameters
        )

        # inserts full schema
        schema_module = inspect.getmodule(self._schema)
        if schema_module is not None:
            schema_source = inspect.getsource(schema_module)
        else:
            raise ValueError("module not found")

        example_json = self._load("examples", "example.json")

        prompt = prompt.replace("{SCHEMA}", schema_source)
        prompt = prompt.replace("{EXAMPLE}", example_json)
        return prompt

    def build_main_prompt(self, parameters: dict[str, str] | None = None) -> str:
        """Build main/inference prompt with schema only.

        Args:
            parameters (dict[str, str] | None): Values for the schema's prompt
                parameters. Defaults to None.

        Returns:
            Complete prompt with parameters and {SCHEMA} replaced

        Raises:
            ValueError: If the parameters are missing, unknown or not accepted by
                this schema.
        """
        prompt = self._substitute_parameters(
            self._load_root("prompt_main.txt"), parameters
        )

        schema_module = inspect.getmodule(self._schema)
        if schema_module is not None:
            schema_source = inspect.getsource(schema_module)
        else:
            raise ValueError("module not found")

        prompt = prompt.replace("{SCHEMA}", schema_source)
        return prompt
