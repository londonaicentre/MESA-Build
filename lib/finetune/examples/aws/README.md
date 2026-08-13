# Fine-tuning on AWS/SageMaker — example

A minimal, runnable reference for LoRA fine-tuning on SageMaker with `HuggingFaceLoRATrainer`, driven by `config.yaml`.
This folder can be used as the starting point for a real pipeline.

## What it shows

The flow is split into two scripts:

- `run_example.py` — prepares + validates a training batch in S3 (`20260123-094248_test-batch`, 10 samples), stages `train.jsonl`, and launches a SageMaker training job. The trainer is wired to a schema + prompt builder (`oncoschema`) describing the target output format, and a standard `config.yaml` holding only training params (the trainer translates it into the SageMaker HuggingFace `hyperparameters` dict).
- `post_process.py` — once the job completes, downloads the trained adapter, merges it with the base model, builds a model card, and uploads to the build bucket under `s3://aicentre-nlpteam-mesa-build/models/<model_name>/<model_name>_<major>_<minor>_<patch>/` as individual files (`*.safetensors`, `config.json`, `tokenizer*`, `model_card.yaml`).

Note that this example passes `push_public=False` to `post_process`. If set `push_public=True`, then this would additionally publish a `.tar.gz` to the public bucket for production runs.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/)
- AWS credentials with access to the `aicentre-nlpteam-mesa-build` bucket and the `SagemakerExecutionRole`

## Run

```bash
cd examples/finetune_aws
uv sync

# Prepare + validate the data, stage it to S3, and launch the (billable) training job.
uv run python run_example.py
```

This downloads + validates the 10-sample batch, uploads `train.jsonl`, and launches a `ml.g5.xlarge` SageMaker job. The job runs asynchronously. `run_example.py` prints the S3 output path and job name, then exits.

Monitor the job in the AWS SageMaker console. Once its status is `Completed`, run the post-processing step with the values printed above:

```bash
uv run python post_process.py \
    --s3-output-path jobs/train/<job_id>/output \
    --job-name mesa-<job_id>-...
```

This downloads the adapter, merges it with the base model, and uploads the unpacked merged model to the build bucket under `models/qwen-onco-example/qwen-onco-example_1_0_0/`.

## Notes

- Running `run_example.py` is billable - it starts a `ml.g5.xlarge` training job.

## License

finetune-aws-example © 2026 by London AI Centre is licensed under CC BY-NC-ND 4.0.
To view a copy of this license, visit <https://creativecommons.org/licenses/by-nc-nd/4.0/>
