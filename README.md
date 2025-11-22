# GenoLlama

Training LLMs for genomic biomarker extraction from NHS genomic laboratory hub reports.

## Repository structure

<img src="_assets/repo.png" alt="isolated" width="500"/>

- `/assets`

    - Pydantic model for specifying an expected LLM output structure
    
    - Prompt templates.

- `/lib/datagen`: Bootstrapping synthetic data for LLM fine-tuning

- `/lib/finetune`: LLM fine-tuning

- `/lib/infer`: LLM inference

- `/lib/utils`: Reusable functions

- `/pipelines`: Individual pipelines for developing text to schema models (e.g. GenoLlama)