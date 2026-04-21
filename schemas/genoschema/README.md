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

## Schema

![Schema overview](https://londonaicentre.github.io/MESA-Build/schemas/genoschema.png)

| Type | Values |
| ---- | ------ |
| ClinicalFindingType | morbidity, patient_finding, family_history |
| TestType | dna, fish, karyotype, pcr, mlpa, other |
| ResultEntityType | chromosome, gene, exon, variant, protein |
| ResultStatus | abnormal, normal, uncertain_significance, failed_or_inconclusive |

## License

This project uses a proprietary license issued by Guy's and St Thomas' NHS Foundation Trust, enabling free (non-commercial) use by NHS organisations. See LICENSE files for details.
