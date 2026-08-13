# Training-data generation on AWS (batch) — example

A minimal reference for generating schema-aligned training data with the `datagen` library using Bedrock batch inference, then uploading it to the build bucket.

## What it shows

The flow is split into three scripts (batch inference is asynchronous):

- `generate.py` — downloads + extracts a document batch from `s3://aicentre-nlpteam-mesa-build/documents/test-batch-2026-01-17-001.tar.gz`, builds a JSONL batch specification, uploads it to S3, and submits a Bedrock batch inference job with `BedrockBatchGenerator`. The job ID is saved to `.job_id.json`.
- `extract.py` — once the job completes, downloads the output from S3 and parses it into `{content, output}` samples in `./data/trainingdata/` with `extract_batch_output`. Outputs that parse but fail schema validation go to `./data/trainingdata/invalid/`.
- `upload.py` — packages the validated samples and uploads them to `s3://aicentre-nlpteam-mesa-build/trainingdata/<timestamp>_docgen-batch-test/` with `TrainingDataUploader.upload`. The system prompt from the schema is baked into the JSONL.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/)
- AWS credentials with access to the `aicentre-nlpteam-mesa-build` bucket and to Bedrock in `eu-west-2`.
- A Bedrock Execution IAM Role ARN with S3 and cross-region model access (set `BEDROCK_EXECUTION_ROLE` in `generate.py`).

## Run

```bash
cd examples/aws/batch
uv sync

# Step 1: build the batch specification and submit the job to Bedrock.
uv run python generate.py

# Step 2: once the job completes (check the AWS console), download and parse outputs.
uv run python extract.py

# Step 3: package + upload the validated samples to the build bucket.
uv run python upload.py
```

## Notes

- Batch inference is substantially cheaper than real-time inference for large volumes, but jobs can take several hours to complete.
- The job ID is persisted in `.job_id.json` so `extract.py` knows where to find the output in S3.

## License

docgen-aws-batch-example © 2026 by London AI Centre is licensed under CC BY-NC-ND 4.0.
To view a copy of this license, visit <https://creativecommons.org/licenses/by-nc-nd/4.0/>
