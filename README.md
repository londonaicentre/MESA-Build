# GenoLlama
Training LLMs for genomic biomarker extraction tool from NHS genomic laboratory hub reports

# Project structure
- `/schemas`: Pydantic model for specifying an expected LLM output structure.
- `/datagen`: File for generating synthetic data for LLM fine-tuning

# Fine-tune data generation
Synthetic data for fine-tuning LLMs is generated using the following components:
(1) `bootstrap.csv`, where each row is an outline for a synthetic genomics report
(2) `/examples`, containing numerous examples of reports and target output schema (json) that conforms to the pydantic model.
(3) `generate.py`, script that passes each bootstrap row to an LLM API, while passing examples and instructions as part of a system prompt, to generate fine-tuning samples