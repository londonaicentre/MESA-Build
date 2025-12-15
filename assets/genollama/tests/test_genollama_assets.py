import re
from re import DOTALL
from typing import Any
from unittest.mock import MagicMock, mock_open

import pytest
from pytest import MonkeyPatch

from genollama_assets.schema import GenomicTestReport
from genollama_assets.wrapper import GenoLlamaAssets


@pytest.fixture(scope="session")
def genollama_assets() -> GenoLlamaAssets:
    return GenoLlamaAssets()


def test_validate_json_examples(genollama_assets: GenoLlamaAssets) -> None:
    # Validate that we can instantiate and validate the schema
    # This would probably fail at an earlier stage if there were issues in reality.
    genollama_assets.schema.model_json_schema()


def test_validate_example_schemas(genollama_assets: GenoLlamaAssets) -> None:
    genollama_assets.validate_json_examples()


def test_load_system_prompt_datagen(genollama_assets: GenoLlamaAssets) -> None:
    systemprompt_datagen: str = genollama_assets.load_system_prompt()
    # contains boilerplate text
    assert (
        "You are a clinical genetics specialist experienced in writing genomic lab reports and extracting genomic information into structured schema."
        in systemprompt_datagen
    )
    # does not contain unwanted markdown
    assert (
        "SYSTEM PROMPT FOR GENOMIC REPORT GENERATION AND EXTRACTION"
        not in systemprompt_datagen
    )
    # contains schema
    assert '{"properties": {"test_subject":' in systemprompt_datagen
    # contains an example
    assert "CYTOGENETICS AND MOLECULAR GENETICS REPORT" in systemprompt_datagen


def test_load_bootstrap_user_prompt(genollama_assets: GenoLlamaAssets) -> None:
    assert "foo bar baz" in genollama_assets.load_bootstrap_user_prompt("foo bar baz")


def test_load_datagen_user_prompt(genollama_assets: GenoLlamaAssets) -> None:
    bootstrap_row: dict[str, Any] = {
        "test_type": "foo",
        "test_details": "",
        "result_entities": "",
        "result_description": "",
        "clinical_context": "",
        "disease_context": "",
        "family_history": "",
        "test_subject": "",
        "clinical_implications": "",
        "recommendations": "bar",
        "report_style": "baz",
    }
    assert re.match(
        r".*foo.*bar.*baz",
        genollama_assets.load_datagen_user_prompt(bootstrap_row),
        flags=DOTALL,
    )
