"""Tests for genoschema package."""

import pytest

from genoschema.prompt_builder import PromptBuilder
from genoschema.schema import GenomicTestReport


def test_validate_schema() -> None:
    """Test that we can instantiate and validate the schema."""
    GenomicTestReport.model_json_schema()


def test_build_datagen_prompt() -> None:
    """Test building data generation prompt."""
    builder = PromptBuilder()
    prompt = builder.build_datagen_prompt()

    # Placeholders replaced
    assert "{SCHEMA}" not in prompt, "Schema placeholder should be replaced"
    assert "{EXAMPLE}" not in prompt, "Example placeholder should be replaced"

    # Schema JSON fields present
    assert "is_genomic_report" in prompt, "Schema should contain key field"
    assert "biomarker_test_results" in prompt or "TestType" in prompt, "Schema should contain test types"

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
    assert "is_genomic_report" in prompt, "Schema should contain key field"
    assert "biomarker_test_results" in prompt or "TestType" in prompt, "Schema should contain test types"
