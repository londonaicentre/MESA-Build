"""Tests for oncoschema package."""

import pytest

from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncoLlamaModel


def test_validate_schema() -> None:
    """Test that we can instantiate and validate the schema."""
    OncoLlamaModel.model_json_schema()


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


def test_validate_json() -> None:
    """Test JSON validation."""
    valid_json = """
    {
        "document_has_primary_cancer_flag": false,
        "primary_cancer_confirmed_flag": false,
        "primary_cancer": null,
        "performance_status": null,
        "other_cancers": null,
        "patient_findings": null,
        "future_plans": null,
        "context_summary": null
    }
    """
    result = PromptBuilder.validate_json(valid_json)
    assert isinstance(result, OncoLlamaModel)
    assert result.document_has_primary_cancer_flag is False
