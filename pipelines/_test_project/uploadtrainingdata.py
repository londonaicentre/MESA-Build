"""
upload_training.py

Example script for uploading training data to S3
"""

import logging
import os
from pathlib import Path

from datagen import TrainingDataUploader
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

logging.basicConfig(level=logging.INFO)

os.environ["AWS_REGION_NAME"] = "eu-west-2"

pb = PromptBuilder()

s3_uri = TrainingDataUploader.upload(
    schema=OncologyModel,
    schema_name="oncoschema",
    system_prompt=pb.build_main_prompt(),
    short_description="test-batch",
    long_description="Test run using fake data batch",
    input_folder=Path("./data/trainingdata"),
)

print(f"Upload complete")
print(f"S3 URI: {s3_uri}")
