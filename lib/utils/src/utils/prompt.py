"""Base prompt builder for schemas"""

import inspect
import re
from abc import ABC
from functools import reduce
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model


class BasePromptBuilder(ABC):
    """Base class for schema prompt builders

    (adapted from SchemaLlamaAssets wrapper)
    """

    __RESERVED_NAMES: frozenset[str] = frozenset({"schema", "example"})
    __NAME_PATTERN: re.Pattern[str] = re.compile(r"[a-z][a-z0-9_]*")
    __VALUE_PATTERN: str = r"^[\w '(),.-]+$"

    def __init__(
        self,
        base_dir: str,
        schema: type[BaseModel],
        parameters: tuple[str, ...] = (),
    ) -> None:
        """Initialize prompt builder.

        Args:
            base_dir: Package name (e.g. 'oncoschema', 'genoschema')
            schema: Pydantic model class for this schema
            parameters (tuple[str, ...]): Names of the caller-supplied prompt
                parameters, each filling its upper-cased placeholder (e.g. `condition`
                fills `{CONDITION}`). Defaults to no parameters.

        Raises:
            ValueError: If a parameter name is not lower snake case or shadows a
                placeholder the builder fills itself.
        """
        if invalid_names := [
            name
            for name in parameters
            if name in BasePromptBuilder.__RESERVED_NAMES
            or not BasePromptBuilder.__NAME_PATTERN.fullmatch(name)
        ]:
            raise ValueError(f"Invalid prompt parameter names: {invalid_names}")
        self._base_dir: Traversable = files(base_dir)
        self._schema: type[BaseModel] = schema
        self.__parameters: tuple[str, ...] = parameters
        fields: dict[str, Any] = {
            name: (
                str,
                Field(pattern=BasePromptBuilder.__VALUE_PATTERN, max_length=100),
            )
            for name in parameters
        }
        self.__parameters_model: type[BaseModel] = create_model(
            "PromptParameters", __config__=ConfigDict(extra="forbid"), **fields
        )

    def get_parameters(self) -> tuple[str, ...]:
        """Get the names of the caller-supplied prompt parameters.

        Returns:
            tuple[str, ...]: The parameter names, empty if the schema takes none.

        """
        return self.__parameters

    def _substitute_parameters(
        self, prompt: str, parameters: dict[str, str] | None
    ) -> str:
        return reduce(
            lambda text, item: text.replace("{" + item[0].upper() + "}", item[1]),
            self.__parameters_model.model_validate(parameters or {})
            .model_dump()
            .items(),
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
                parameters. Defaults to None.

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
