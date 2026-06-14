"""Test example.json validates against schema."""

import json

from importlib.resources import files

from paedacuteschema.schema import PaedAcuteSchemaModel


def test_example_validates() -> None:
    """Test that example.json output validates against PaedAcuteSchemaModel."""
    # Load example
    example_path = files("paedacuteschema").joinpath("examples/example.json")
    example_data = json.loads(example_path.read_text())

    # Validate output against schema
    output = example_data["output"]
    validated = PaedAcuteSchemaModel.model_validate(output)

    # Basic checks
    assert validated.is_clinical_document is True
    assert validated.document_content is not None
    assert validated.entities is not None
    assert len(validated.entities) > 0
