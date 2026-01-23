"""Tests for oncoschema package."""


from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel


def test_validate_schema() -> None:
    """Test that we can instantiate and validate the schema."""
    OncologyModel.model_json_schema()


def test_build_datagen_prompt() -> None:
    """Test building data generation prompt."""
    builder = PromptBuilder()
    prompt = builder.build_datagen_prompt()

    # Placeholders replaced
    assert "{SCHEMA}" not in prompt, "Schema placeholder should be replaced"
    assert "{EXAMPLE}" not in prompt, "Example placeholder should be replaced"

    # Schema JSON fields present
    assert "document_has_primary_cancer_flag" in prompt, "Schema should contain key field"
    assert "TopographyType" in prompt or "primary_cancer" in prompt, "Schema should contain cancer types"

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

    # Schema JSON fields present
    assert "document_has_primary_cancer_flag" in prompt, "Schema should contain key field"
    assert "TopographyType" in prompt or "primary_cancer" in prompt, "Schema should contain cancer types"
