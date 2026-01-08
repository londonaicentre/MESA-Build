"""Prompt builder for oncology schema."""

from oncoschema.schema import OncologyModel
from utils.prompt import BasePromptBuilder


class PromptBuilder(BasePromptBuilder):
    """Build prompts for oncology extraction."""

    def __init__(self) -> None:
        super().__init__("oncoschema", OncologyModel)
