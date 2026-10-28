"""Tests for opnoteschema package."""

from opnoteschema.prompt_builder import PromptBuilder
from opnoteschema.schema import OperationNote


def test_validate_schema() -> None:
    """Test that we can instantiate and validate the schema."""
    OperationNote.model_json_schema()


def test_build_datagen_prompt() -> None:
    """Test building data generation prompt."""
    builder = PromptBuilder()
    prompt = builder.build_datagen_prompt()

    # Placeholders replaced
    assert "{SCHEMA}" not in prompt, "Schema placeholder should be replaced"
    assert "{EXAMPLE}" not in prompt, "Example placeholder should be replaced"

    # Schema source present
    assert "OperationNote" in prompt, "Schema should contain the root model"
    assert "procedures" in prompt, "Schema should contain key field"

    # Example should be present
    assert '"content"' in prompt or "'content'" in prompt, (
        "Example should contain 'content' field"
    )
    assert '"output"' in prompt or "'output'" in prompt, (
        "Example should contain 'output' field"
    )


def test_build_main_prompt() -> None:
    """Test building main prompt."""
    builder = PromptBuilder()
    prompt = builder.build_main_prompt()

    # Placeholders replaced
    assert "{SCHEMA}" not in prompt, "Schema placeholder should be replaced"
    assert "{EXAMPLE}" not in prompt, "Main prompt should not have example placeholder"

    # Schema source present
    assert "OperationNote" in prompt, "Schema should contain the root model"
    assert "procedures" in prompt, "Schema should contain key field"
