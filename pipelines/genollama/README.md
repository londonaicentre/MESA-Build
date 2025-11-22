# GenoLlama

Training LLMs for genomic biomarker extraction from NHS genomic laboratory hub reports.

## Getting started

### Prerequisites

- [python](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Configuration

Within a `.env` file, specify:

```
BEDROCK_API_KEY=
BEDROCK_EXECUTION_ROLE=
BUCKET=
```

### Assets

- `genollama_assets/genollama_assets_types` > `GenomicTestReport`, containing the target schema as a pydantic model.

- `genollama_assets/examples`, containing numerous examples of reports and target output schema (in json) that conforms to the schema.

- `genollama_assets/prompts.py`

### Installation

```
uv venv
source .venv/bin/activate
uv sync
```

## Usage

### Bootstrap file generation

1. Run `bootstrap` with the following arguments:

    - `-m, --model_name`: Name of model to use, eg 'sonnet4' or 'opus4', must have corresponding configuration in the `config.json` file. Defaults to `sonnet4`.

    - `-i, --instruction`: Tailor the bootstrap file output, e.g. point the batch at a type of test, a disease area, a particular proband pattern, a report style, or any other variable. Required.

### Generate synthetic data samples

1. Run `datagen` with the following arguments:

    - `-m, --model_name`: Name of model to use, eg 'sonnet4' or 'opus4', must have corresponding configuration in the `config.json` file. Defaults to `sonnet4`.

    - `-s, --sample_size`: Required number of samples to be generated, defaults to `10`.

    - `-b, --bootstrap`: Path to the bootstrap file (must be in the same directory or provide absolute path). Defaults to `bootstrap.csv`.

    - `-f, --backfill`: Whether to generate additional samples and backfill for missed indices from bootstrap file. Defaults to `False`.

For example `datagen sonnet4 15` will generate 15 samples in the model's subfolder under `samples`, while `datagen sonnet4 23 -f True` will generate sample reports for any missed indices and at least 8 additional samples.

