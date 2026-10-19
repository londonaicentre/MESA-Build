# mesa-build/pipelines

Notebooks for manual end-to-end fine-tuning

## Creating a new pipeline

### Prerequisites

- Linux/Mac
- [uv](https://docs.astral.sh/uv/)

### Quickstart

1. Create a new pipeline boilerplate directory: `uvx copier copy _pipeline <new pipeline name> --trust`.

2. Answer the questions posed.

3. Add custom content to each notebook and config file.

## Correcting training data

Run `/add-trainingdata-correction-config <pipeline>` in Claude Code to create a pipeline's `prompts/correction.txt` and `rules/correction.yaml`, which `BatchOutputCorrector` uses to correct its batch outputs.
