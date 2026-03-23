from typing import Literal

from pydantic import BaseModel


class TrainingMessage(BaseModel):
    """Single message in OpenAI fine-tuning format."""

    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None = None


class TrainingSample(BaseModel):
    """Complete training sample for OpenAI fine-tuning."""

    messages: list[TrainingMessage]
