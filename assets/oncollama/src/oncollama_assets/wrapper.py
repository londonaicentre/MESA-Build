from importlib.resources.abc import Traversable
import inspect
from typing import Any

from pydantic import BaseModel

from schemallama_types.assets.wrapper import SchemaLlamaAssets
from schemallama_types.assets.profile import Profile, Profiles
from oncollama_assets import schema


class OncoLlamaAssets(SchemaLlamaAssets):
    def __init__(self) -> None:
        super().__init__("oncollama_assets")

    # schema
    def validate_schema(
        self, schema: type[BaseModel]
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Validate a schema (pydantic -> dict) with
            additional checks

        Args:
            schema: (type[BaseModel]): The schema to validate

        Returns:
            tuple: The validation result, result description,
                and json version of the schema

        """
        result: bool
        message: str
        json_schema: dict[str, Any] | None
        try:
            result, message, json_schema = super().validate_schema(schema)
            if result and json_schema is not None:
                print(f"Properties: {len(json_schema.get('properties', {}))}")
                print(f"Definitions: {len(json_schema.get('$defs', {}))}")
                return True, "Schema validation successful", json_schema
            else:
                raise ValueError(message)
        except Exception as e:
            return False, f"Schema validation failed: {e}", None

    # prompts
    def load_system_prompt(self, file: str = "systemprompt_infer.md") -> str:
        """Create a system prompt

        Args:
            file (str, optional): The template file to use for the system prompt.
                Defaults to the inference system prompt.

        Returns:
            str: The system prompt

        """
        schema_content: str = inspect.getsource(schema)
        system_prompt_template: str = self._load("prompts", file)
        return system_prompt_template.replace("{SCHEMA}", schema_content)

    def load_bootstrap_user_prompt(self, instructions: str) -> str:
        """TODO: Implement upon the use of bootstrapping in oncollama."""
        return ""

    def load_datagen_user_prompt(self, row: dict[str, Any]) -> str:
        return row["content"]

    # profiles
    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        cancer_profiles: list[Profile] = []
        cancer_type: str = file_path.name
        profiles: Profiles = Profiles(file_path)
        for profile_id, profile_data in profiles.items.items():
            cancer_profiles.append(
                Profile(
                    **{
                        "profile_id": profile_id,
                        "morphology": profile_data.morphology or "UNKNOWN",
                        "descriptive_name": profile_data.descriptive_name or "",
                        "biomarker_profile": profile_data.biomarker_profile or "",
                        "cancer_type": cancer_type,
                        "source_file": file_path.name,
                    }
                )
            )
        return cancer_profiles

    def format_profile_prompt(self, profile: Profile) -> str:
        """Create the profile portion of a docsynth user prompt

        Args:
            profile (Profile): The profile object containing the information
                to include in the prompt

        Returns:
            str: The profile prompt

        """
        lines: list[str] = ["## USE THIS PRIMARY CANCER PROFILE"]
        lines.append("")
        lines.append(
            f"**Primary Diagnosis that should appear verbatim in document:** {profile.descriptive_name}"
        )
        lines.append("")
        lines.append(
            f"**Biomarker Profile - Note that these are the ONLY molecular biomarker results that should be given for this patient:** {profile.biomarker_profile}"
        )
        lines.append("")
        return "\n".join(lines)
