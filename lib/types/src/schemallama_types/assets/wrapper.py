from abc import ABC, abstractmethod
from importlib.resources import files
from importlib.resources.abc import Traversable
import json
from pathlib import Path
from typing import Any, cast

from jsonschema import ValidationError
from pydantic import BaseModel

from schemallama_types.assets.profile import Profile
from schemallama_types.assets.sampling import Content, Style


class SchemaLlamaAssets(ABC):
    def __init__(self, base_dir: str) -> None:
        self._base_dir: Traversable = files(base_dir)

    def _load(self, folder: str, file: str) -> str:
        return self._base_dir.joinpath(f"{folder}/{file}").read_text()

    # schema
    def validate_json(
        self, json_str: str, schema: type[BaseModel]
    ) -> tuple[bool, str, dict[str, Any] | None]:
        try:
            parsed: dict[str, Any] = json.loads(json_str)
            schema(**parsed)
            return True, "Valid", parsed
        except json.JSONDecodeError as e:
            return False, f"JSON Parse Error: {e}", None
        except ValidationError as e:
            return False, f"Schema Validation Error: {e}", None
        except Exception as e:
            return False, f"Validation Error: {e}", None

    def validate_schema(
        self, schema: type[BaseModel]
    ) -> tuple[bool, str, dict[str, Any] | None]:
        try:
            json_schema: dict[str, Any] = schema.model_json_schema()
            return True, "Schema validation successful", json_schema
        except Exception as e:
            return False, f"Schema validation failed: {e}", None

    # prompts
    def load_user_prompt_template(self, template_name: str) -> str:
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

    # profiles
    @abstractmethod
    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        pass

    @abstractmethod
    def format_profile_prompt(self, profile: Profile) -> str:
        pass

    def load_all_profiles(self) -> list[Profile]:
        all_profiles: list[Profile] = []
        items: list[Traversable] = cast(
            list[Traversable], sorted(self._base_dir.joinpath("profiles").iterdir())
        )
        item: Traversable
        for item in items:
            if item.is_file() and item.name.endswith(".yml"):
                profiles: list[Profile] = self._load_profiles_from_file(item)
                all_profiles.extend(profiles)
        return all_profiles

    def load_profiles_from_files(self, filenames: list[str]) -> list[Profile]:
        all_profiles: list[Profile] = []
        for filename in filenames:
            file_path: str = "profiles/" + filename
            try:
                profiles: list[Profile] = self._load_profiles_from_file(
                    self._base_dir.joinpath(file_path)
                )
                all_profiles.extend(profiles)
            except FileNotFoundError:
                raise FileNotFoundError(f"Profile file not found: {file_path}")
        return all_profiles

    # styles
    def load_style_data(self) -> Style:
        return Style(self._base_dir.joinpath("style.yml"))

    def load_content_data(self) -> Content:
        return Content(self._base_dir.joinpath("content.yml"))

    # structures
    def load_structures(self, enabled_structures: list[str]) -> dict[str, str]:
        self.structures: dict[str, str] = {}
        for filename in enabled_structures:
            file_path: str = "structure/" + filename
            try:
                self.structures[filename] = self._base_dir.joinpath(
                    file_path
                ).read_text()
            except FileNotFoundError:
                raise FileNotFoundError(f"Structure file not found: {file_path}")
        return self.structures

    def get_structure_name_without_extension(self, filename: str) -> str:
        return Path(self._base_dir.joinpath("structure/" + filename).name).stem
