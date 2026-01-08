"""Test that the canonical example validates."""

from oncoschema.prompt_builder import PromptBuilder


def test_example_validates():
    """Test that the canonical example validates against the schema."""
    builder = PromptBuilder()
    validated = builder.validate_example()
    assert validated is not None
    assert validated.document_has_primary_cancer_flag is not None
