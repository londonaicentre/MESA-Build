"""Tests for paedacuteschema package."""

from paedacuteschema.prompt_builder import PromptBuilder
from paedacuteschema.schema import PaedAcuteSchemaModel


def test_validate_schema() -> None:
    """Test that we can instantiate and validate the schema."""
    PaedAcuteSchemaModel.model_json_schema()


def test_build_datagen_prompt() -> None:
    """Test building data generation prompt."""
    builder = PromptBuilder()
    prompt = builder.build_datagen_prompt()

    # Placeholders replaced
    assert "{SCHEMA}" not in prompt, "Schema placeholder should be replaced"
    assert "{EXAMPLE}" not in prompt, "Example placeholder should be replaced"

    # Schema fields present
    assert "is_clinical_document" in prompt, "Schema should contain key field"
    assert "AcuteSignType" in prompt or "entity" in prompt, "Schema should contain acute signs"

    # Example should be present
    assert '"content"' in prompt or "'content'" in prompt, "Example should contain 'content' field"
    assert '"output"' in prompt or "'output'" in prompt, "Example should contain 'output' field"


def test_build_main_prompt() -> None:
    """Test building main prompt."""
    builder = PromptBuilder()
    prompt = builder.build_main_prompt()

    # Placeholders replaced
    assert "{SCHEMA}" not in prompt, "Schema placeholder should be replaced"
    assert "{EXAMPLE}" not in prompt, "Main prompt should not have example placeholder"

    # Schema fields present
    assert "is_clinical_document" in prompt, "Schema should contain key field"
    assert "AcuteSignType" in prompt or "entity" in prompt, "Schema should contain acute signs"
