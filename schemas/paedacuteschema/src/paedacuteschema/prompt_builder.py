"""Prompt builder for paediatric acute triage schema."""

from paedacuteschema.schema import PaedAcuteSchemaModel
from utils.prompt import BasePromptBuilder


class PromptBuilder(BasePromptBuilder):
    """Build prompts for paediatric acute triage sign extraction."""

    def __init__(self) -> None:
        super().__init__("paedacuteschema", PaedAcuteSchemaModel)
