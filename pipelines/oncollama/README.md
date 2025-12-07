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
```

### Installation

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Usage

### Synthetic file generation

1. Run `docsynth`.

### Bootstrap file generation

1. Reformat docsynth outputs as a bootstrap file for datagen. Run `bootstrap`.

### Generate synthetic data samples

1. Run `datagen` with the following arguments:

    - `-m, --model_name`: Name of model to use, eg 'sonnet4' or 'opus4', must have corresponding configuration in the `config.json` file. Defaults to `sonnet4`.

    - `-s, --sample_size`: Required number of samples to be generated, defaults to `10`.

    - `-b, --bootstrap`: Path to the bootstrap file (must be in the same directory or provide absolute path). Defaults to `bootstrap.csv`.

    - `-f, --backfill`: Whether to generate additional samples and backfill for missed indices from bootstrap file. Defaults to `False`.
