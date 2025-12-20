from typing import Any

from schemallama_types.assets.wrapper import SchemaLlamaAssets
from oncollama_assets.schema import OncoLlamaModel


class OncoLlamaAssets(SchemaLlamaAssets):
    def __init__(self) -> None:
        super().__init__("oncollama_assets", OncoLlamaModel)

    def load_system_prompt(self, file: str = "systemprompt_infer.md") -> str:
        """Create a system prompt

        Args:
            file (str, optional): The template file to use for the system prompt.
                Defaults to the inference system prompt.

        Returns:
            str: The system prompt

        """
        system_prompt_template: str = self._load("prompts", file)
        return system_prompt_template.replace(
            "{SCHEMA}", str(self._schema.model_json_schema())
        )

    def load_bootstrap_user_prompt(self, instructions: str) -> str:
        """TODO: Implement upon the use of bootstrapping in OncoLlama."""
        raise NotImplementedError("Bootstrapping not yet implemented for OncoLlama.")

    def load_datagen_user_prompt(self, row: dict[str, Any]) -> str:
        return row["content"]
