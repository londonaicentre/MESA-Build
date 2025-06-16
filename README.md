# GenoLlama
Training LLMs for genomic biomarker extraction tool from NHS genomic laboratory hub reports

# Project structure
- `/schema`: Pydantic model for specifying an expected LLM output structure.
- `/datagen`: Bootstrapping synthetic data for LLM fine-tuning

# Fine-tune data generation
Synthetic data for fine-tuning LLMs is generated using the following components:
(1) `bootstrap.csv`, where each row is an outline for a synthetic genomics report
(2) `_scratch/examples`, containing numerous examples of reports and target output schema (in json) that conforms to the pydantic model.
(3) `schema/genomicextractmode.py`, containing the target schema that should appear as a json, containing the extract for each synthetic report.

# TO DO
`generate.py`, script that passes each bootstrap row to an LLM API, while passing examples and instructions as part of a system prompt, to generate fine-tuning samples