"""Test that the canonical example validates."""

from mesa_types import TrainingExample
from genoschema.prompt_builder import PromptBuilder
from genoschema.schema import GenomicTestReport


def test_example_validates():
    """Test that the canonical example validates against the schema."""
    builder = PromptBuilder()

    example_str = builder._load("examples", "example.json")
    example_data = TrainingExample.model_validate_json(example_str)

    validated = GenomicTestReport.model_validate(example_data.output)

    assert validated is not None
    assert validated.is_genomic_report is not None
