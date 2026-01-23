"""
document.py

Data model for document JSONs
"""

from pydantic import BaseModel


class Document(BaseModel):
    """
    This is the standard format for documents in MESA documents,
    used as input for training data generation.

    Attributes:
        content: document text content
        source: descriptive source identifier (e.g., "docsynth-oncology", "wikipedia-paragraphs")
        timestamp: creation timestamp
    """

    content: str
    source: str
    timestamp: str
