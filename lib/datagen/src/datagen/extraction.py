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

from pydantic import BaseModel

from mesa_types import Document
from utils.llm import LLM


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


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract JSON from LLM response.

    Args:
        response: LLM response text

    Returns:
        Parsed JSON dict if successful, None otherwise

    """
    logger = logging.getLogger(__name__)

    # try to parse response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # try to find JSON in a code block
        extracted: bool
        content: str
        extracted, _, content = LLM.extract_output_content(response)
        if extracted:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

        # if fails, try to find any JSON-like structure
        json_match = re.search(r"{[\s\S]*}", response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                raise json.JSONDecodeError("Could not parse JSON from response", "", 0)
        logger.error("No valid JSON found in response")
        return None


def validate_with_pydantic(
    output_data: dict[str, Any], schema: type[BaseModel]
) -> tuple[bool, BaseModel | None]:
    """Validate output against Pydantic schema.

    Args:
        output_data: JSON data to validate
        schema: Pydantic schema class

    Returns:
        Tuple of (is_valid, validated_model)

    """
    logger = logging.getLogger(__name__)
    try:
        validated_report: BaseModel = schema(**output_data)
        return True, validated_report
    except Exception as e:
        logger.error(f"Pydantic validation error: {e}")
        return False, None


def extract_validate_and_save_sample(
    response: str,
    source: str,
    content: str,
    schema: type[BaseModel],
    schema_name: str,
    schema_version: str,
    output_folder: str,
) -> bool:
    """Extract sample from model response, validate it and save it.

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
    logger = logging.getLogger(__name__)

    extracted_json: dict[str, Any] | None = extract_json_from_response(response)

    #print for debugging
    print(f"LLM returned JSON: {json.dumps(extracted_json, indent=2)[:500] if extracted_json else 'None'}")
    if extracted_json is not None:
        if not isinstance(extracted_json, dict):
            logger.error("Extracted output is not a dictionary")
            return False

        # vs schema
        is_valid: bool
        validated_output: BaseModel | None
        is_valid, validated_output = validate_with_pydantic(extracted_json, schema)

        if is_valid and validated_output is not None:
            # build training example
            json_output = {
                "content": content,
                "output": validated_output.model_dump()
            }

            doc = Document(content=content, source=source, timestamp="")
            output_filename: str = os.path.join(
                output_folder, get_output_filename(schema_name, schema_version, doc)
            )
            try:
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)
                logger.info(f"Successfully saved output to {output_filename}")
                return True
            except Exception as e:
                logger.error(f"Error saving JSON to file: {e}")
                return False
        else:
            logger.error("Pydantic validation failed")

            # save invalid
            doc = Document(content=content, source=source, timestamp="")
            debug_filename: str = os.path.join(
                output_folder,
                f"invalid_{get_output_filename(schema_name, schema_version, doc)}",
            )
            debug_output = {
                "content": content,
                "output": extracted_json
            }
            with open(debug_filename, "w", encoding="utf-8") as f:
                json.dump(debug_output, f, indent=4, ensure_ascii=False)
            return False
    else:
        logger.warning("Skipping file save due to JSON parsing failure")
        logger.debug(
            "LLM response:",
            response,
        )
        return False
