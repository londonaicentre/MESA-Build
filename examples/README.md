## MESA-Build Examples

Standalone, runnable projects that are the canonical demonstrations of data generation and finetune flow (and also represent smoke tests).

- `examples/docgen_aws/` — real-time training-data generation with `LLMGenerator` + `oncoschema`, then upload to the build bucket.
- `examples/docgen_gemini/` — same flow as `docgen_aws` but inference runs against Google Gemini via LiteLLM, with the API key loaded from a local `.env` (documents are still pulled from S3).
- `examples/finetune_aws/` — SageMaker / `HuggingFaceLoRATrainer` fine-tuning.
- `examples/finetune_mlx/` — local `MLXLoRATrainer` fine-tuning (depends on `mesa-finetune[mlx]`).