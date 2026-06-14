# Training-data generation on Google Gemini — example

A minimal reference for generating schema-aligned training data with the `datagen` library and the `oncoschema` schema package, running inference against a Google Gemini endpoint (Google AI Studio) routed through LiteLLM.

## What it shows

The flow is split into two scripts:

- `generate.py` — downloads + extracts a document batch from `s3://aicentre-nlpteam-mesa-build/documents/test-batch-2026-01-17-001.tar.gz`, then runs real-time Gemini inference over each document with `LLMGenerator`. The Gemini key is loaded from `.env` via `load_dotenv()`. Validated `{content, output}` samples are written to `./data/trainingdata/`; outputs that parse but fail schema validation go to `./data/trainingdata/invalid/` and are excluded downstream.
- `upload.py` — packages the validated samples and uploads them to `s3://aicentre-nlpteam-mesa-build/trainingdata/<timestamp>_docgen-gemini-example-batch-06-2026/` with `TrainingDataUploader.upload`. The system prompt from the schema is baked into the JSONL.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/)
- A Google AI Studio (Gemini) API key.
- AWS credentials with access to the `aicentre-nlpteam-mesa-build` bucket in `eu-west-2`. Documents are still downloaded from S3, and the upload step writes to S3 — so AWS access is required even though inference runs on Gemini.

## Run

```bash
cd examples/docgen_gemini
uv sync

# Provide your Gemini key. load_dotenv() reads .env from the current working
# directory, so run the scripts from this folder.
cp .env.example .env
# then edit .env so it reads:  GEMINI_API_KEY=<your-google-ai-studio-key>

# Step 1: generate samples from the document batch (billable - real-time Gemini calls;
# documents are pulled from S3, so AWS credentials are needed).
uv run python generate.py

# Step 2: package + upload the validated samples to the build bucket (needs AWS).
uv run python upload.py
```

## Notes

- Running `generate.py` is billable — it makes one real-time Gemini inference call per document.
- Run scripts from this directory so `load_dotenv()` finds the local `.env`.
