"""Test that the canonical example validates."""

from mesa_types import TrainingExample

from conditionschema.prompt_builder import PromptBuilder
from conditionschema.schema import ConditionModel


def test_example_validates() -> None:
    """Test that the canonical example validates against the schema."""
    builder = PromptBuilder()

    example_str = builder._load("examples", "example.json")
    example_data = TrainingExample.model_validate_json(example_str)

    validated = ConditionModel.model_validate(example_data.output)

    assert len(validated.mentions) == 4
