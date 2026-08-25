"""Prompt builder for Histopathology schema."""

from utils.prompt import BasePromptBuilder

from oncopathschema.schema import OncoPathModel


class PromptBuilder(BasePromptBuilder):
    """Build prompts for Histopathology extraction."""

    def __init__(self) -> None:
        super().__init__("oncopathschema", OncoPathModel)
