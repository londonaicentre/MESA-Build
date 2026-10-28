"""Prompt builder for Condition schema."""

from utils.prompt import BasePromptBuilder

from conditionschema.schema import ConditionModel


class PromptBuilder(BasePromptBuilder):
    """Build prompts for Condition extraction."""

    def __init__(self) -> None:
        super().__init__("conditionschema", ConditionModel, ("condition", "cui"))
