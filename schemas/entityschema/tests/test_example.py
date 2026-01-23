"""Test example.json validates against schema."""

import json

import pytest
from importlib.resources import files

from entityschema.schema import EntitySchemaModel


def test_example_validates() -> None:
    """Test that example.json output validates against EntitySchemaModel."""
    # Load example
    example_path = files("entityschema").joinpath("examples/example.json")
    example_data = json.loads(example_path.read_text())

    # Validate output against schema
    output = example_data["output"]
    validated = EntitySchemaModel.model_validate(output)

    # Basic checks
    assert validated.is_clinical_document is True
    assert validated.document_content is not None
    assert validated.entities is not None
    assert len(validated.entities) > 0
