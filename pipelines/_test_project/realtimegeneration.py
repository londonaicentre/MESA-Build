"""
realtimegeneration.py

Example of project using real-time generation for training data
Outputs are saved to ./data/trainingdata/
"""

import logging
import os

from datagen import LLMGenerator
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

logging.basicConfig(level=logging.INFO)

os.environ["AWS_REGION_NAME"] = "eu-west-2"

pb = PromptBuilder()

gen = LLMGenerator(
    system_prompt=pb.build_datagen_prompt(),
    user_prompt_function=lambda doc: doc['content'],
    schema=OncologyModel,
    schema_name="oncoschema",
    model_name="bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    api_key="",  # does not need if IAM credentials
    document_batches=["test-batch-2026-01-17-001.tar.gz"],
)

gen.generate(10)
print("Generation completed")
