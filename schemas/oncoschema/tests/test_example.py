"""Test that the canonical example validates."""

from mesa_types import TrainingExample
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel


def test_example_validates() -> None:
    """Test that the canonical example validates against the schema."""
    builder = PromptBuilder()

    example_str = builder._load("examples", "example.json")
    example_data = TrainingExample.model_validate_json(example_str)

    validated = OncologyModel.model_validate(example_data.output)

    assert validated is not None
    assert validated.document_has_primary_cancer_flag is not None
