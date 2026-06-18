"""Prompt builder for Operation notes schema."""

from opnoteschema.schema import OperationNote
from utils.prompt import BasePromptBuilder


class PromptBuilder(BasePromptBuilder):
    """Build prompts for Operation notes extraction."""

    def __init__(self) -> None:
        super().__init__("opnoteschema", OperationNote)
