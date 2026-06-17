"""
extract.py

Step 2: once the batch job started by generate.py has completed, download and
parse its outputs into ./data/trainingdata/.
"""

import logging

from datagen import BedrockBatchGenerator
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

logging.basicConfig(level=logging.INFO)

BUCKET = "<bucket_name>"

DOCUMENT_BATCHES = ["test-batch-2026-01-17-001.tar.gz"]

pb = PromptBuilder()

gen = BedrockBatchGenerator(
    system_prompt=pb.build_datagen_prompt(),
    user_prompt_function=lambda doc: doc["content"],
    schema=OncologyModel,
    schema_name="oncoschema",
    model_name="sonnet4",
    document_batches=DOCUMENT_BATCHES,
)

successful, failed = gen.extract_batch_output(bucket=BUCKET)
print(f"Extraction complete: {successful} successful, {failed} failed")
print("Samples in ./data/trainingdata/")
