# Fine-tuning locally with MLX — example

A minimal, runnable reference for LoRA fine-tuning on local Apple Silicon with `MLXLoRATrainer`, driven by `config.yaml`.
This folder can be used as the starting point for a real pipeline.

## What it shows

`run_example.py` wires the trainer to:

- a **schema + prompt builder** (`oncoschema`) that describe the target output format, and
- a **training batch** in S3 (`20260123-094248_test-batch`, 10 samples),
- a **standard `config.yaml`** holding only training params (the trainer translates it into the `mlx_lm`-native run-config).

It then runs the full flow end-to-end:

- prepares + validates the data locally,
- trains a LoRA adapter via `mlx_lm.lora`,
- fuses the adapter into the base model via `mlx_lm.fuse`,
- builds a model card and uploads the unpacked merged model to the build bucket.

The merged model is published to the build bucket under `s3://aicentre-nlpteam-mesa-build/models/<model_name>/<model_name>_<major>_<minor>_<patch>/` as individual files (`*.safetensors`, `config.json`, `tokenizer*`, `model_card.yaml`).

The example passes `push_public=False` to `post_process`. If set `push_public=True`, then this would additionally publish a `.tar.gz` to the public bucket for production runs.

## Prerequisites

- Apple Silicon (macOS, arm64) - the `mlx` extra only installs there.
- [`uv`](https://docs.astral.sh/uv/)
- AWS credentials with access to the `aicentre-nlpteam-mesa-build` bucket (for the upload)

## Run

```bash
cd examples/finetune_mlx
uv sync                      # installs mlx + mlx-lm too (Apple Silicon only, via the [mlx] extra)

uv run python run_example.py
```

This prepares the data, trains a LoRA adapter locally, fuses it into the base model, and uploads the unpacked merged model to the build bucket under `models/qwen-onco-mlx-example/qwen-onco-mlx-example_1_0_0/`. The fused model is left locally at `data/models/qwen-onco-mlx-example/target/*.safetensors`.