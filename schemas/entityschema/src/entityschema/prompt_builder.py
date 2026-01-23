"""Prompt builder for entity schema."""

from entityschema.schema import EntitySchemaModel
from utils.prompt import BasePromptBuilder


class PromptBuilder(BasePromptBuilder):
    """Build prompts for clinical entity extraction."""

    def __init__(self) -> None:
        super().__init__("entityschema", EntitySchemaModel)
