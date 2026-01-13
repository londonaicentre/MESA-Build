# MESA Build

Training LLMs for biomarker extraction from unstructured NHS documents.

## Pipelines

Current pipelines in this repository:

- [OncoLlama](pipelines/oncollama/): Generating high fidelity synthetic cancer letters, and fine-tuning LLMs for structured data extraction.

- [GenoLlama](pipelines/genollama/): Training LLMs for genomic biomarker extraction from NHS genomic laboratory hub reports.

## Repository structure

```text
📁 SCHEMA_LLAMA
├── pipelines/             # Individual pipelines for developing text to schema models
├── assets/                # Static per-pipeline assets
├── lib/                   # Reusable functionality across pipelines
├──── docsynth/            # Configuration driven unstructured document generation
├──── datagen/             # Bootstrapping synthetic data for LLM fine-tuning
├──── finetune/            # LLM fine-tuning
├──── infer/               # LLM inference
├────── demo/llamadeploy   # Deploy a Llama model on AWS SageMaker AI
├────── demo/deploy        # Deploy LiteLLM proxy for SageMaker AI Llama models
├────── local/deploy       # Deploy model weight distribution infrastructure for llamaserve
├────── local/llamaserve   # Serve llama models locally
├──── types/               # Cross-project type definitions
├──── utils/               # Reusable functions
└── README.md              # Project overview and documentation
```

<img src="_assets/repo.png" alt="isolated" width="500"/>

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
