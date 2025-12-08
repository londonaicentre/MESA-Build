from litellm import Choices, Message, ModelResponse
import pytest


@pytest.fixture
def model_response() -> ModelResponse:
    message: Message = Message(content="The quick brown fox jumped over the lazy dog")
    choice: Choices = Choices(message=message)
    return ModelResponse(
        choices=[choice],
    )
