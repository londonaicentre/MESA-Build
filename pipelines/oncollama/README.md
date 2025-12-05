# OncoLlama

Generating high fidelity synthetic cancer letters, and fine-tuning LLMs for structured data extraction

## Getting started

### Prerequisites

- [python](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Configuration

Within a `.env` file, specify:

```
llm__anthropic__api_key=
llm__local__model=
```

### Installation

```
uv venv
source .venv/bin/activate
uv sync
```

## Usage

### Synthetic file generation

1. Run `docsynth`.