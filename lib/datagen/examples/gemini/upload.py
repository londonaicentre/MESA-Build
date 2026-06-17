"""
upload.py

Step 2: package the validated samples in ./data/trainingdata/ and upload them
to s3://aicentre-nlpteam-mesa-build/trainingdata/<timestamp>_<short_description>/.

TrainingDataUploader re-validates each local sample against the schema, then
writes three artifacts to the run folder:
  - train_openai_<short_description>.jsonl  (OpenAI messages format; the
        `build_main_prompt()` inference prompt is baked in as the system message)
  - metadata.yaml                           (schema, schema_version, descriptions,
        num_samples, created_at)
  - train_openai_<short_description>_samples.tar.gz  (raw per-sample JSONs)
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
    system_prompt=pb.build_main_prompt(),  # main (inference) prompt, baked into JSONL
    short_description="docgen-gemini-example-batch-06-2026",
    long_description="Example oncoschema training data (Gemini) from test-batch-2026-01-17-001",
    input_folder=Path("./data/trainingdata"),
    # bucket / s3_prefix / region default to
    # aicentre-nlpteam-mesa-build / trainingdata / eu-west-2
)

print("Upload complete")
print(f"S3 URI: {s3_uri}")
