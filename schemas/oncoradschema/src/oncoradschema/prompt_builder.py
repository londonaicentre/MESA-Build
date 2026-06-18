"""Prompt builder for Radiology schema."""

from oncoradschema.schema import OncoRadModel
from utils.prompt import BasePromptBuilder


class PromptBuilder(BasePromptBuilder):
    """Build prompts for Radiology extraction."""

    def __init__(self) -> None:
        super().__init__("oncoradschema", OncoRadModel)
