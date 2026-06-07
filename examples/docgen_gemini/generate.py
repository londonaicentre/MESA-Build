"""
generate.py

Step 1: generate oncoschema training samples from a document batch in S3
using real-time LLM inference against Google Gemini (Google AI Studio).

Documents are still downloaded from S3, so AWS credentials are required in
addition to a Gemini API key. Only the inference calls go to Gemini.
"""

import logging
import os

from datagen import LLMGenerator
from dotenv import load_dotenv
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

logging.basicConfig(level=logging.INFO)

# Loads GEMINI_API_KEY from a local .env file
load_dotenv()

# AWS is still needed here: LLMGenerator downloads + extracts the document batch
# from s3://aicentre-nlpteam-mesa-build/documents/ in its constructor. Only the
# inference calls below go to Gemini.
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
    model_name="gemini/gemini-3-flash-preview",
    api_key="",  # leave empty: LiteLLM reads GEMINI_API_KEY from env (loaded by load_dotenv)
    document_batches=DOCUMENT_BATCHES,  # downloaded + extracted in constructor (needs AWS)
    temperature=1.0,  # official recommendation for gemini 3 family as minimum temp
    max_tokens=32768,
)

gen.generate(SAMPLE_SIZE)
print("Generation complete. Samples in ./data/trainingdata/")
