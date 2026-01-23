# Genoschema

Schema package for genomic biomarker extraction from NHS genomic laboratory hub reports.

## Structure

```text
📁 genoschema
├── examples/            # Training examples showing document input and structured output
├── schema.py            # Pydantic model for specifying expected output structure
├── prompt_builder.py    # Prompt builder for data generation and inference
├── prompt_datagen.txt   # Prompt template with example (for training data generation)
├── prompt_main.txt      # Prompt template without example (for inference/deployment)
└── py.typed             # Type checking marker
```

## Usage

```python
from genoschema.prompt_builder import PromptBuilder

# Initialize builder
builder = PromptBuilder()

# Build data generation prompt (with example)
datagen_prompt = builder.build_datagen_prompt()

# Build main/inference prompt (without example)
main_prompt = builder.build_main_prompt()
```

## License

This project uses a proprietary license issued by Guy's and St Thomas' NHS Foundation Trust (see [LICENSE.md](LICENSE.md)).

The contents of this repository are designed for NHS organisations to use on private data.
