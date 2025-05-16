# GenoLlama
genetic biomarker extraction tool using Llama models


# schemas and examples
`schemas` Contains a Pydantic schema for specifying the expected LLM output structure, while `examples` contains the expected 'output' following the schema based on the deidentified input 'content'.

The `validator.py` script can be run from within the `schemas` folder to ensure example files adhere to the specified `GenomicTestReport` schema.