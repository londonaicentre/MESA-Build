# OncoLlama

Generating high fidelity synthetic cancer letters, and fine-tuning LLMs for structured data extraction

## Getting started

### Prerequisites

- [python](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Configuration

Within a `.env` file, specify:

```text
llm__anthropic__api_key=
llm__local__model=
BEDROCK_EXECUTION_ROLE=
BUCKET=
US_BUCKET=
SAGEMAKER_EXECUTION_ROLE=
INSTANCE_TYPE=
IMAGE=
```

### Installation

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Usage

### Synthetic oncology note generation

1. Run `docsynth`.

### Bootstrap file generation

1. Reformat docsynth outputs as a bootstrap file for datagen. Run `bootstrap`.

### Generate fine-tuning input data

1. Run `datagen` with the following arguments:

    - `-m, --model_name`: Name of model to use, eg 'sonnet4' or 'opus4', must have corresponding configuration in the `config.json` file. Defaults to `sonnet4`.

    - `-s, --sample_size`: Required number of samples to be generated, defaults to `10`.

    - `-f, --backfill`: Whether to generate additional samples and backfill for missed indices from bootstrap file. Defaults to `False`.

### Start an OncoLlama fine-tuning run

1. Run `finetune` with the following arguments:

    - `-f, --file`: Name of the AWS Bedrock Anthropic batch inference output file containing sample data to use as input. Defaults to `anthropic_batch_job.jsonl.out`.

    - `-d, --dry_run`: Whether to simulate calling AWS endpoints. Defaults to `False`.

### Deploy OncoLlama to AWS

1. Run `deploy` with the following arguments:

    - `-p, --path`: Path within S3 bucket to the zipped weights of the model to deploy. Required.

    - `[command]`: The action to perform: `up` (deploy) or `down` (delete). Required.
