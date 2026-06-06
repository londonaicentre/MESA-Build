# Architecture

**This is living documentation and should be updated with each merge**

MESA-Build is a monorepo for fine-tuning LLMs to perform **M**edical **E**ntity **E**xtraction with **S**chema **A**lignment. It produces models that read clinical documents and emit structured JSON conforming to a defined Pydantic schema.

The repository is organised as a set of independently versioned Python packages. Three groups exist: shared libraries (`lib/`), extraction schemas (`schemas/`), and per-model pipelines (`pipelines/`). Infrastructure and CI live in `deploy/` and `.github/workflows/`.

## End-to-end flow

The framework moves data through three stages, mirrored by three S3 prefixes in the shared bucket `aicentre-nlpteam-mesa-build`:

1. **Documents** (`documents/`) — raw clinical text, packaged as `Document` JSON objects in `.tar.gz` batches.
2. **Training data** (`trainingdata/`) — schema-conformant examples generated from documents by a teacher LLM, stored as OpenAI-format JSONL batches.
3. **Models** (`models/`) — fine-tuned model artifacts and their model cards.

The pipeline is:

```
raw documents (S3 documents/)
   -> datagen: teacher LLM extracts structured output per schema
   -> training data (S3 trainingdata/, OpenAI JSONL)
   -> finetune: LoRA fine-tuning on SageMaker
   -> model artifacts + model card (S3 models/)
```

Datagen and finetune both validate every record against the schema's Pydantic model, so schema alignment is enforced at generation time and again before training.

## Components

### lib/types (`londonaicentre-mesa-types`)

The shared data-model layer with no dependencies on other internal packages. Defines the core Pydantic types passed between stages:

- `Document` — standard input record (`content`, `source`, `timestamp`).
- `TrainingExample` — a generated example (`content`, `output` dict).
- `ModelCard` — metadata for a finetuned model, including base model, version, training data references, and the output schema; serialisable to YAML. It resolves the schema package name and version at runtime via `importlib.metadata`.

Everything else depends on this package.

### lib/utils (`londonaicentre-mesa-utils`)

Reusable functionality shared by schemas and the data/training libraries. Depends only on `mesa-types`. Key modules:

- `llm.py` — `LLM` wrapper over LiteLLM for real-time completions, plus Pydantic models for batch inputs/outputs.
- `aws.py` — `AWS` helper for S3 upload/download and constructing Anthropic Bedrock batch-inference entries.
- `prompt.py` — `BasePromptBuilder`, the abstract base each schema subclasses. It loads `prompt_datagen.txt` / `prompt_main.txt` from the schema package and injects the schema source and an example to produce generation and inference prompts.
- `assets.py` — markdown-to-text helper.

Schema packages depend on utils for `BasePromptBuilder`.

### lib/datagen (`londonaicentre-mesa-datagen`)

Training-data generation. Depends on `mesa-types` and `mesa-utils`. Turns document batches into validated training examples. Key modules:

- `document_loader.py` — `DocumentLoader` downloads and extracts document batches from S3 and validates each as a `Document`.
- `llm_generator.py` — `LLMGenerator` runs real-time generation through any LiteLLM-supported provider.
- `batch_generator.py` — `BedrockBatchGenerator` uses the AWS Bedrock batch-inference API for large-scale, lower-cost generation.
- `extraction.py` — parses and validates LLM output against the schema and writes training samples.
- `trainingdata_uploader.py` — `TrainingDataUploader` packages samples as OpenAI-format JSONL and uploads to S3 `trainingdata/`.
- `version_detector.py` — resolves the installed schema package version, embedded into output filenames for provenance.
- `config.py` / `config/config.json` — named model configs (e.g. `sonnet4`, `opus4`) mapping to Bedrock model IDs and regions.

Both generators take a schema class, a prompt (from a schema's prompt builder), and a list of document batches.

### lib/finetune (`londonaicentre-mesa-finetune`)

LLM fine-tuning. Depends on `mesa-utils` and `mesa-datagen`. Key modules:

- `trainingdata_handler.py` — `TrainingDataHandler` downloads training batches from S3, validates them against the schema and expected system prompt, and concatenates them into a single `train.jsonl`.
- `hf_estimator.py` — `HuggingFaceLoRATrainer` orchestrates LoRA fine-tuning as a SageMaker HuggingFace estimator job, taking the schema, prompt builder, training batches, hyperparameters, and AWS config. It also produces a `ModelCard`.
- `scripts/train_lora.py` — the training entry point uploaded to and run on the SageMaker GPU instance, using `transformers`, `trl`, `peft`, and `datasets` to run SFT with a LoRA adapter.

### schemas/

Model-agnostic extraction schema packages, each independently versioned and published to PyPI. Each depends only on `pydantic` and `mesa-utils`. They define *what* to extract; the libraries define *how*. Each package follows the same layout:

- `schema.py` — Pydantic model defining the expected output structure.
- `prompt_builder.py` — `PromptBuilder` subclass of `BasePromptBuilder` bound to the schema.
- `prompt_datagen.txt` — prompt template with an example, for training-data generation.
- `prompt_main.txt` — prompt template without an example, for inference/deployment.
- `examples/example.json` — example document input and structured output.

Schema packages:

- `oncoschema` — oncology / cancer clinical documents.
- `genoschema` — NHS genomic laboratory hub biomarker reports.
- `entityschema` — general clinical entity extraction. Unlike the others it does not vendor `mesa-utils` as an editable local source.
- `oncoradschema` — oncology radiology; work in progress (schema source only, no package metadata yet).

### pipelines/

Per-model projects that compose a schema with the datagen and finetune libraries to produce a specific fine-tuned model. These are the concrete consumers of everything above; they are not published. Each pipeline pins the libraries and one schema as editable local sources.

- `genollama` — notebook-driven pipeline (`datagen.ipynb`, `finetune.ipynb`) combining `genoschema` with datagen and finetune.
- `oncoqwen` — Qwen-based oncology model; holds generated training data.
- `_example_train` — minimal example wiring `oncoschema` to `HuggingFaceLoRATrainer` (`orchestrate.py`).
- `_test_project` — example scripts for real-time generation (`realtimegeneration.py`) and training-data upload (`uploadtrainingdata.py`).
- `_scratch` — local experimentation space (model merges, MLX exports).

A typical pipeline:

```python
from datagen import LLMGenerator, TrainingDataUploader
from finetune import HuggingFaceLoRATrainer
from oncoschema.prompt_builder import PromptBuilder
from oncoschema.schema import OncologyModel

pb = PromptBuilder()
# generate -> upload -> fine-tune, all keyed on OncologyModel + pb prompts
```

## Dependency graph

```
types  <-  utils  <-  schemas (oncoschema, genoschema, entityschema)
   ^         ^
   |         |
   +----  datagen  <-  finetune
                          ^
pipelines  ---------------+  (also depend on one schema)
```

Direction of arrows is "depended on by". `types` is the foundation; `utils` builds on it; `datagen` and `finetune` build the generation/training stack; schemas plug in via `utils`; pipelines sit on top and bind one schema to the stack.

## Infrastructure

- `deploy/main.tf` — Terraform for the S3 bucket `aicentre-nlpteam-mesa-build` (versioned, with `documents/`, `trainingdata/`, `models/` prefixes) and a public bucket `aicentre-nlpteam-mesa-public`. State is stored in S3.
- Bedrock is the default inference backend; SageMaker runs the fine-tuning jobs. Default region is `eu-west-2`.

## Conventions and CI

- Every library and schema is a standalone package built with setuptools and managed with `uv`. Internal dependencies are wired as editable local sources via `[tool.uv.sources]` for development, and resolved from PyPI for published packages.
- Each package has a `Makefile`: `make test` runs `mypy` and `pytest`; `make prettier` runs `ruff` check/format.
- Packages are versioned with SemVer. Changes are reviewed via PR and merged to `main`.
- `.github/workflows/` holds per-package CI: `*-test.yml` runs ruff, mypy, and pytest on PRs touching that package; `*-build-and-publish.yml` tags and publishes to PyPI when a PR to `main` merges.
- `assets-build-docs.yml` generates an entity-relationship diagram from each schema and publishes schema README/docs to GitHub Pages.
- The repo-wide `CHANGELOG.md` records cross-package breaking changes.
