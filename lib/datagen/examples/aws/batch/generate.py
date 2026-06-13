"""
generate.py

Step 1: submit a Bedrock batch inference job for oncoschema training data
generation. Once submitted, wait for the job to complete, then run extract.py.
"""

import logging

from datagen import BedrockBatchGenerator
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

logging.basicConfig(level=logging.INFO)

############################################################################################
# Supply the ARN of a Bedrock Execution IAM Role with S3 and model access, and
# the name of an S3 bucket to hold the batch specification and outputs.
BEDROCK_EXECUTION_ROLE = "arn:aws:iam::<account_id>:role/<role_name>"
BUCKET = "<bucket_name>"

DOCUMENT_BATCHES = ["test-batch-2026-01-17-001.tar.gz"]
SAMPLE_SIZE = 10

pb = PromptBuilder()

gen = BedrockBatchGenerator(
    system_prompt=pb.build_datagen_prompt(),
    user_prompt_function=lambda doc: doc["content"],
    schema=OncologyModel,
    schema_name="oncoschema",
    model_name="sonnet4",
    document_batches=DOCUMENT_BATCHES,
)

job_id = gen.generate_via_batch(
    sample_size=SAMPLE_SIZE,
    bedrock_execution_role=BEDROCK_EXECUTION_ROLE,
    bucket=BUCKET,
)
print(f"Batch job submitted: {job_id}")
print("Wait for the job to complete in the AWS console, then run extract.py.")
