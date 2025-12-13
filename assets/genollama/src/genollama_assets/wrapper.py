from importlib.resources.abc import Traversable
import json
from typing import Any

from litellm import cast
from pydantic import BaseModel, ValidationError

from schemallama_types.assets.wrapper import SchemaLlamaAssets
from schemallama_types.assets.profile import Profile
from genollama_assets.schema import GenomicTestReport
from utils.assets import Assets


class GenoLlamaAssets(SchemaLlamaAssets):
    def __init__(self) -> None:
        super().__init__("genollama_assets")

    # schema
    def validate_json_examples(self, schema: type[BaseModel]) -> tuple[bool, str]:
        """Validate stored genomic example json files (str -> dict)

        Args:
            schema: (type[BaseModel]): The schema to validate against

        Returns:
            tuple: The validation result and result description

        """
        items: list[Traversable] = cast(
            list[Traversable], sorted(self._base_dir.joinpath("examples").iterdir())
        )
        item: Traversable
        result: bool
        message: str
        parsed: dict[str, Any] | None
        for item in items:
            if item.is_file() and item.name.endswith(".json"):
                result, message, parsed = super().validate_json(
                    json.dumps(json.loads(item.read_text())["output"]), schema
                )
                try:
                    if result and parsed is not None:
                        loaded_example: BaseModel = schema(**parsed)
                        loaded_example.model_dump_json()
                    else:
                        raise ValueError(message)
                except (ValidationError, ValueError) as e:
                    return False, f"Example {item.name} failed validation: {e}"
        return True, "All examples checked"

    def validate_schema(
        self, schema: type[BaseModel]
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Validate a genomic schema (pydantic -> dict),
            and output the result.

        Args:
            schema: (type[BaseModel]): The schema to validate

        Returns:
            tuple: The validation result, result description,
                and json version of the schema

        """
        result: bool
        message: str
        json_schema: dict[str, Any] | None
        result, message, json_schema = super().validate_schema(schema)
        if result and json_schema is not None:
            with open("schema.json", "w") as output_file:
                json.dump(json_schema, output_file, indent=4)
        return result, message, json_schema

    # prompts
    def load_system_prompt(self, file: str = "systemprompt_datagen.md") -> str:
        """Create a system prompt

        Args:
            file (str, optional): The template file to use for the system prompt.
                Defaults to the datagen system prompt.

        Returns:
            str: The system prompt

        """
        schema_content: str = json.dumps(GenomicTestReport.model_json_schema())
        system_prompt_template: str = self._load("prompts", file)
        examples_path: str = "examples"
        e1: str = ""
        e2: str = ""
        e3: str = ""
        e4: str = ""
        try:
            e1 = self._load(examples_path, "e1.json")
            e2 = self._load(examples_path, "e2.json")
            e3 = self._load(examples_path, "e3.json")
            e4 = self._load(examples_path, "e4.json")
        except FileNotFoundError as e:
            print(f"Warning: Could not load example file: {e}")
        return Assets.markdown_to_text(
            system_prompt_template.replace("{schema_content}", schema_content)
            .replace("{e1}", e1)
            .replace("{e2}", e2)
            .replace("{e3}", e3)
            .replace("{e4}", e4)
        )

    def load_bootstrap_user_prompt(self, instructions: str) -> str:
        """Create a user prompt for bootstrap file generation

        Args:
            instructions (str): Instruction to tailor the bootstrap file output

        Returns:
            str: The user prompt

        """
        user_prompt: str = f"""Please now generate 20 rows according to the above instructions as a CSV file. These rows should {instructions}. While conforming to these instructions, please also ensure that rows are varied, and represent a range of different report types and styles."""
        return user_prompt

    def load_datagen_user_prompt(self, row: dict[str, Any]) -> str:
        """Create a user prompt for sample file generation

        Args:
            row (dict): The bootstrap file row from which to generate data to tailor the sample

        Returns:
            str: The user prompt

        """
        user_prompt: str = f"""Please generate a genomic laboratory report based on the following test scenario:

            Test Type: {row["test_type"]}
            Test Details: {row["test_details"]}
            Result Entities: {row["result_entities"]}
            Result Description: {row["result_description"]}
            Clinical Context: {row["clinical_context"]}
            Disease Context: {row["disease_context"]}
            Family History: {row["family_history"]}
            Test Subject: {row["test_subject"]}
            Clinical Implications: {row["clinical_implications"]}
            Recommendations: {row["recommendations"]}
            Report Style: {row["report_style"]}

            Generate a realistic genomic laboratory report incorporating all these details.
            Then extract the information into the structured schema format."""
        return user_prompt

    # profiles
    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        """TODO: Implement upon the use of docsynth profiles in genollama."""
        return []

    def format_profile_prompt(self, profile: Profile) -> str:
        """TODO: Implement upon the use of docsynth profiles in genollama."""
        return ""
