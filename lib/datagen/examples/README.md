## MESA-Build Datagen Examples

Standalone, runnable projects that are the canonical demonstrations of data generation (and also represent smoke tests).

- `aws/llm/` — real-time training-data generation with `LLMGenerator` + `oncoschema`, then upload to the build bucket.
- `aws/batch/` — same flow but using `BedrockBatchGenerator` for cost-effective async batch inference via the Bedrock batch API.
- `gemini/` — same flow as `docgen_aws` but inference runs against Google Gemini via LiteLLM, with the API key loaded from a local `.env` (documents are still pulled from S3).