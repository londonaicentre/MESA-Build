"""
extraction.py

Output extraction/processing helpers
Used for real-time LLM
"""

import hashlib
import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from mesa_types import Document

logger = logging.getLogger(__name__)


def get_output_filename(
    schema_name: str, schema_version: str, doc: Document
) -> str:
    """Generate metadata-embedded output filename for a document.

    Args:
        schema_name: Schema name for filename
        schema_version: Schema version for filename
        doc: Document to generate filename for

    Returns:
        Filename in format: {schema_name}{version}_{source}_{content_hash}.json

    """
    content_hash = hashlib.md5(doc.content.encode()).hexdigest()[:8]
    return f"{schema_name}{schema_version}_{doc.source}_{content_hash}.json"


def _try_parse_and_validate(
    text: str, schema: type[BaseModel]
) -> tuple[BaseModel | None, dict[str, Any] | None]:
    """Try to parse text as JSON and validate against schema.

    Args:
        text: Text to parse as JSON
        schema: Pydantic schema for validation

    Returns:
        Tuple of (validated_model, extracted_data)
    """
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None, None
        validated = schema.model_validate(data)
        return validated, data
    except (json.JSONDecodeError, ValidationError):
        return None, None


def _save_json_file(data: dict[str, Any], filepath: str) -> bool:
    """Save JSON data to file.

    Args:
        data: Data to save
        filepath: Full file path

    Returns:
        True if successful

    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def extract_and_validate_json(
    response: str, schema: type[BaseModel]
) -> tuple[BaseModel | None, dict[str, Any] | None]:
    """Extract JSON from response and validate against Pydantic schema.

    Stepwise:
    1. Content between <output> tags (lowercase)
    2. Take whole response
    3. Largest JSON-like object (greedy regex)

    At each stage, validates both JSON parsing and Pydantic schema.
    """
    # extract from <output> tags
    output_match = re.search(r"<output>([\s\S]*?)</output>", response)
    if output_match:
        validated, data = _try_parse_and_validate(output_match.group(1).strip(), schema)
        if validated:
            return validated, data
        if data:
            return None, data

    # try whole response
    validated, data = _try_parse_and_validate(response.strip(), schema)
    if validated:
        return validated, data
    if data:
        return None, data

    # greedy regex fallback
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        validated, data = _try_parse_and_validate(json_match.group(0), schema)
        if validated:
            return validated, data
        if data:
            return None, data

    logger.error("No valid JSON found in response")
    return None, None


def save_training_sample(
    response: str,
    source: str,
    content: str,
    schema: type[BaseModel],
    schema_name: str,
    schema_version: str,
    output_folder: str,
) -> bool:
    """Save training sample from LLM response.

    Args:
        response: Raw response from the model
        source: Document source for filename
        content: Document content for filename hash
        schema: Pydantic schema class
        schema_name: Schema name for filename
        schema_version: Schema version for filename
        output_folder: Output folder path

    Returns:
        Whether the operations were successful

    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "invalid"), exist_ok=True)

    validated_output, extracted_data = extract_and_validate_json(response, schema)

    if extracted_data:
        print(f"LLM returned JSON: {json.dumps(extracted_data, indent=2)[:500]}")
    else:
        print("LLM response: No valid JSON extracted")

    doc = Document(content=content, source=source, timestamp="")
    base_filename = get_output_filename(schema_name, schema_version, doc)

    # success case
    if validated_output is not None:
        json_output = {"content": content, "output": validated_output.model_dump()}
        output_filepath = os.path.join(output_folder, base_filename)
        return _save_json_file(json_output, output_filepath)

    # fail case: schema validation
    if extracted_data is not None:
        logger.error("Validation failed")
        debug_output = {"content": content, "output": extracted_data}
        debug_filepath = os.path.join(output_folder, "invalid", base_filename)
        _save_json_file(debug_output, debug_filepath)
        return False

    # fail case: no json
    logger.warning("Skipping due to JSON extraction failure")
    logger.debug(f"LLM response: {response[:500]}")
    return False
