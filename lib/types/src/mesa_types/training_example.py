"""Standard training example document type."""

from typing import Any

from pydantic import BaseModel


class TrainingExample(BaseModel):
    """Standard structure for schema training examples."""

    content: str
    output: dict[str, Any]
