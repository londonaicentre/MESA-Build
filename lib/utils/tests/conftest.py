import pytest

from utils.llm import Choice, ModelResponse, ResponseMessage


@pytest.fixture
def model_response() -> ModelResponse:
    return ModelResponse(
        choices=[
            Choice(
                message=ResponseMessage(
                    content="The quick brown fox jumped over the lazy dog"
                ),
                finish_reason="foo",
            )
        ]
    )
