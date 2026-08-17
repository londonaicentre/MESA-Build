# MESA-Build

End-to-end framework for fine-tuning LLMs for **M**edical **E**ntity **E**xtraction with **S**chema **A**lignment.

<img src="_assets/repo.png" alt="repo" width="500"/>

Other parts of the MESA family include:
- [**MESA**](https://github.com/londonaicentre/MESA) - tutorials and example notebooks showing end-to-end model development and inference
- **MESA-DocSynth** - configuration driven synthetic document generation
- **MESA-Deploy** - deployment of inference endpoint to local or cloud infrastructure
- **MESA-Runner** - stateless runtime for deploying models to live infrastructure
- **MESA-Ansible** - ansible projects for provisioning entire MESA infrastructure
- **MESA-Validate** - simple Python application for HITL validation of MESA model outputs vs source documents


## Overview

MESA-Build is an internal team monorepo for generating training data and fine-tuning LLMs, supporting:

- Schema design and release
- Training data generation and registration
- Model fine-tuning and registration
- Bedrock integration by default

MESA has an open license for the NHS through Guy's and St. Thomas' Hospitals, permitting use of software framework and developed models without restriction.

## Repository Structure

```text
📁 MESA-Build
├── deploy/              # Terraform for AWS S3 setup  
├── lib/                 # Reusable functionality
│   ├── datagen/         # Training data generation
│   ├── finetune/        # LLM fine-tuning
│   ├── types/           # Shared definitions
│   └── utils/           # Utility functions (e.g. LiteLLM, AWS)
├── pipelines/           # Notebooks for manual end-to-end fine-tuning
├── runners/             # CLI wrappers (runners) for automating pipeline functionality
├── schemas/             # Domain specified schemas
└── .github/workflows/   # CI/CD automation
```

## AWS Buckets

The pipeline persists data and artifacts in two S3 buckets. The **build** bucket is the working store for training data and built models; the **public** bucket is an opt-in distribution target for finished models.

| Bucket | Role |
|---|---|
| `aicentre-nlpteam-mesa-build` | Synthetic documents + training data + built (unpacked) models |
| `aicentre-nlpteam-mesa-public` | Public distribution of finished models (tarballs) |

Region is `eu-west-2` throughout (default arg in the utils; passed via `aws_config["region"]` from the trainers).


## Project Flow

0. **Documents** are packaged as `Document` JSON objects in `.tar.gz` batches. Documents can be manually uploaded, or produced from synthetic document pipelines (outside of MESA-Build):

   ```text
   s3://aicentre-nlpteam-mesa-build/documents
   ```

1. **Training-data generation** (`lib/datagen`) results in upload of training data batches to the build bucket:

   ```text
   s3://aicentre-nlpteam-mesa-build/trainingdata/<batch_name>/<batch_name>.jsonl
   ```

2. **Fine-tuning** (`lib/finetune`) takes a set of batch(es) and produces a merged model:

   - HF / SageMaker (`HuggingFaceLoRATrainer`) will stage `train.jsonl` to S3, and reads/writes job artefacts under the build bucket:

     ```text
     s3://aicentre-nlpteam-mesa-build/jobs/train/<job_id>/input/train.jsonl
     s3://aicentre-nlpteam-mesa-build/jobs/train/<job_id>/output/...     # adapter from SageMaker
     ```

   - MLX (`MLXLoRATrainer`) trains and fuses locally and does not stage any files to S3. Each run works under `data/models/<model_name>/<job_id>/`. A transient GPU abort (Metal "innocent victim") is retried automatically from the latest local checkpoint; after retries are exhausted the folder is kept so the job can be continued with `--resume --spec <spec>` once the machine is fixed. On success or cancellation the folder is removed. Resume reads checkpoints from local disk, so it assumes the same runner; mlx-lm is pinned because resume depends on its checkpoint/iters semantics.

3. **Publish** (`post_process` on either trainer) — the primary target is the build bucket:

   ```text
   s3://aicentre-nlpteam-mesa-build/models/<model_name>/<model_name>_<major>_<minor>_<patch>/
       config.json, *.safetensors, tokenizer*, model_card.yaml      # individual files, no tarball
   ```

   With `push_public=True`, a tarball is additionally pushed to the public bucket:

   ```text
   s3://aicentre-nlpteam-mesa-public/<model_name>/<model_name>_<major>_<minor>_<patch>.tar.gz
       (model files + model_card.yml + LICENSE.md)
   ```

## Schema Packages

[**oncoschema**](schemas/oncoschema/src/oncoschema/schema.py): Pydantic schema and prompts for extracting information from cancer clinical documents

[**genoschema**](schemas/genoschema/src/genoschema/schema.py): Pydantic schema and prompts for extracting biomarker information from NHS genomic laboratory hub reports.

[**entityschema**](schemas/entityschema/src/entityschema/schema.py): Schema package for clinical entity extraction from NHS medical documents.

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
