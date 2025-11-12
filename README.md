# GenoLlama

Training LLMs for genomic biomarker extraction from NHS genomic laboratory hub reports.

## Repository structure

<img src="_assets/repo.png" alt="isolated" width="500"/>

- `/assets`

    - Pydantic model for specifying an expected LLM output structure
    
    - Prompt templates.

- `/datagen`: Bootstrapping synthetic data for LLM fine-tuning

- `/finetune`: LLM fine-tuning

- `/infer`: LLM inference

- `/utils`: Reusable functions