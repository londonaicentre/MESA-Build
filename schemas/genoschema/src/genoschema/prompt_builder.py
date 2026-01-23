"""Prompt builder for genomic schema."""

from genoschema.schema import GenomicTestReport
from utils.prompt import BasePromptBuilder


class PromptBuilder(BasePromptBuilder):
    """Build prompts for genomic report extraction."""

    def __init__(self) -> None:
        super().__init__("genoschema", GenomicTestReport)
