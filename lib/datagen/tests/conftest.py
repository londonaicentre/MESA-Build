from typing import Any

import pytest
from litellm import ModelResponse, Choices, Message

from tests.types import TestSchema
from tests.claude import TestSampleGenerator


@pytest.fixture(scope="session")
def valid_sample_json_output() -> str:
    return '{"content": "foo", "output": {"foo": {"foo": "bar"}, "baz": "qux", "qux": [1, 2, 3]}}'


@pytest.fixture
def valid_model_response(valid_sample_json_output: str) -> ModelResponse:
    message: Message = Message(
        content="<OUTPUT>" + valid_sample_json_output + "</OUTPUT>"
    )
    choice: Choices = Choices(message=message)
    return ModelResponse(
        choices=[choice],
    )


@pytest.fixture(scope="session")
def sample_generator() -> TestSampleGenerator:
    def test_user_prompt(param: dict[str, Any]) -> str:
        return "bar"

    return TestSampleGenerator(
        "foo", test_user_prompt, TestSchema, "sonnet4", "qux.csv", "foobar"
    )
