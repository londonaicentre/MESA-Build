# MESA-Build

MESA = and end-to-end framework for fine-tuning LLMs for **M**edical **E**ntity **E**xtraction with **S**chema **A**lignment.

## Overview

MESA-Build is an internal team monorepo for generating training data and fine-tuning LLMs, supporting:
- Schema design and release
- Training data generation and registration
- Model fine-tuning and registration
- Bedrock integration by default

## Repository Structure

```text
📁 MESA-Build
├── schemas/               # Schema packages (model-agnostic)
├── lib/                  # Reusable functionality
│   ├── datagen/         # Training data generation
│   ├── finetune/        # LLM fine-tuning
│   ├── types/           # Shared definitions
│   └── utils/           # Utility functions
├── pipelines/           # Pipeline code for each model
└── .github/workflows/   # CI/CD automation
```

## Schema Packages

**oncoschema**: Pydantic schema and prompts for extracting information from cancer clinical documents

**genoschema**: Pydantic schema and prompts for extracting biomarker information from NHS genomic laboratory hub reports.

Each schema package is independently versioned and published to PyPI

## Development

Most folders are self-contained python packages, allowing for the selective release of assets and pipeline logic. Under this setup, a development/update cycle might look like the following:

- Install dependencies (`uv sync`)

- Add/modify features to/in source code

- Add/update tests for these features

- Run formatter (`make prettier`)

- Run type check and unit tests (`make test`)

- Increment package version according to SemVer

- Commit changes to a feature branch

- Create a PR for the feature branch, and, following CI success and approval, merge with main

- Create a tag for the merge commit referencing the version increment

- Build and push the package to pypi (`python -m build` and `uv run twine upload -r pypi dist/*`), or rely on CI/CD for this

## License

This project uses a proprietary license issued by Guy's and St Thomas' NHS Foundation Trust, enabling free (non-commercial) use by NHS organisations. See LICENSE files for details.
