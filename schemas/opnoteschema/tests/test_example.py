"""Test that the canonical example validates."""

from mesa_types import TrainingExample

from opnoteschema.prompt_builder import PromptBuilder
from opnoteschema.schema import OperationNote


def test_example_validates() -> None:
    """Test that the canonical example validates against the schema."""
    builder = PromptBuilder()

    example_str = builder._load("examples", "example.json")
    example_data = TrainingExample.model_validate_json(example_str)

    validated = OperationNote.model_validate(example_data.output)

    assert validated is not None
    assert validated.is_operation_note is True
