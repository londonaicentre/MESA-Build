"""Tests for conditionschema package."""

import pytest
from pydantic import ValidationError

from conditionschema.prompt_builder import PromptBuilder
from conditionschema.schema import ConditionModel


def test_validate_schema() -> None:
    """Test that we can instantiate and validate the schema."""
    ConditionModel.model_json_schema()


def test_build_datagen_prompt() -> None:
    """Test building data generation prompt."""
    prompt = PromptBuilder().build_datagen_prompt(
        {"condition": "foo bar", "cui": "123456"}
    )

    assert "{CONDITION}" not in prompt
    assert "{CUI}" not in prompt
    assert "{SCHEMA}" not in prompt
    assert "{EXAMPLE}" not in prompt
    assert "the condition foo bar (SNOMED CT 123456)" in prompt
    assert "ConditionModel" in prompt
    assert '"output"' in prompt


def test_build_main_prompt() -> None:
    """Test building main prompt."""
    prompt = PromptBuilder().build_main_prompt(
        {"condition": "foo bar", "cui": "123456"}
    )

    assert "{CONDITION}" not in prompt
    assert "{CUI}" not in prompt
    assert "{SCHEMA}" not in prompt
    assert "{EXAMPLE}" not in prompt
    assert "the condition foo bar (SNOMED CT 123456)" in prompt
    assert "ConditionModel" in prompt


@pytest.mark.parametrize(
    "parameters",
    [None, {"condition": "foo"}, {"condition": "foo\nbar", "cui": "123456"}],
)
def test_build_main_prompt_invalid_parameters_raises(
    parameters: dict[str, str] | None,
) -> None:
    """Test that missing or disallowed parameters are rejected."""
    with pytest.raises(ValidationError):
        PromptBuilder().build_main_prompt(parameters)
