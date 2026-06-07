## MESA-Build Examples

Standalone, runnable projects that are the canonical demonstrations of data generation and finetune flow (and also represent smoke tests).

- `examples/docgen_aws/` — real-time training-data generation with `LLMGenerator` + `oncoschema`, then upload to the build bucket.
- `examples/finetune_aws/` — SageMaker / `HuggingFaceLoRATrainer` fine-tuning.
- `examples/finetune_mlx/` — local `MLXLoRATrainer` fine-tuning (depends on `mesa-finetune[mlx]`).