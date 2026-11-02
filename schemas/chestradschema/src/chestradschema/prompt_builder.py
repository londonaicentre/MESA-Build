"""Prompt builder for Chest radiology schema."""

from utils.prompt import BasePromptBuilder

from chestradschema.schema import ChestRadModel


class PromptBuilder(BasePromptBuilder):
    """Build prompts for Chest radiology extraction."""

    def __init__(self) -> None:
        super().__init__("chestradschema", ChestRadModel)
