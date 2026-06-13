"""
generate.py

Step 1: generate oncoschema training samples from a document batch in S3
using real-time LLM inference.
"""

import logging
import os

from datagen import LLMGenerator
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

logging.basicConfig(level=logging.INFO)

os.environ["AWS_REGION_NAME"] = "eu-west-2"

############################################################################################    
# `document_batches` is a list, so you can pass several batches at once, e.g.:
#
#     DOCUMENT_BATCHES = [
#         "test-batch-2026-01-17-001.tar.gz",
#         "test-batch-2026-01-17-002.tar.gz",
#     ]
#
# Each batch is downloaded from s3://aicentre-nlpteam-mesa-build/documents/ and
# extracted into its own ./data/documents/<batch_name>/ folder, then all their
# documents are concatenated into one ordered list (batch order, then sorted
# filename within each batch). SAMPLE_SIZE caps the total across all batches and
# is consumed in sequence from the start of that list (not randomly).
DOCUMENT_BATCHES = ["test-batch-2026-01-17-001.tar.gz"]
SAMPLE_SIZE = 10

pb = PromptBuilder()

gen = LLMGenerator(
    system_prompt=pb.build_datagen_prompt(),  # schema + worked example injected
    user_prompt_function=lambda doc: doc["content"],
    schema=OncologyModel,
    schema_name="oncoschema",  # uses the installed londonaicentre-oncoschema version
    model_name="bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    api_key="",  # supply if provided with a Bedrock API key, otherwise leave this empty as AWS credentials read from env
    document_batches=DOCUMENT_BATCHES,  # downloaded + extracted in constructor
    temperature=0.001,  # near-deterministic; raise for more diverse samples
    max_tokens=32768,
)

gen.generate(SAMPLE_SIZE)
print("Generation complete. Samples in ./data/trainingdata/")
