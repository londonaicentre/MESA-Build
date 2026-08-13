# Training-data generation on AWS — example

A minimal reference for generating schema-aligned training data with the `datagen` library and the `oncoschema` schema package, then uploading it to the build bucket.

## What it shows

The flow is split into two scripts:

- `generate.py` — downloads + extracts a document batch from `s3://aicentre-nlpteam-mesa-build/documents/test-batch-2026-01-17-001.tar.gz`, then runs real-time Bedrock inference (`claude-sonnet-4-5`) over each document with `LLMGenerator`. Validated `{content, output}` samples are written to `./data/trainingdata/`; outputs that parse but fail schema validation go to `./data/trainingdata/invalid/` and are excluded downstream.
- `upload.py` — packages the validated samples and uploads them to `s3://aicentre-nlpteam-mesa-build/trainingdata/<timestamp>_docgen-test/` with `TrainingDataUploader.upload`. The system prompt from the schema is baked into the JSONL.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/)
- AWS credentials with access to the `aicentre-nlpteam-mesa-build` bucket and to Bedrock in `eu-west-2`.

## Run

```bash
cd examples/docgen_aws
uv sync

# Step 1: generate samples from the document batch (billable - real-time Bedrock calls).
uv run python generate.py

# Step 2: package + upload the validated samples to the build bucket.
uv run python upload.py
```

## Notes

- Running `generate.py` is billable — it makes one real-time Bedrock inference call per document.
- `generate()` is resumable: documents whose deterministic output file already exists are skipped, so re-running only fills in missing samples.

## License

docgen-aws-example © 2026 by London AI Centre is licensed under CC BY-NC-ND 4.0.
To view a copy of this license, visit <https://creativecommons.org/licenses/by-nc-nd/4.0/>
