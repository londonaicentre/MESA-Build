# SchemaLlama

Training LLMs for biomarker extraction from unstructured NHS documents.

## Repository structure

```
📁 SCHEMA_LLAMA
├── pipelines/             # Individual pipelines for developing text to schema models
├── assets/                # Static per-pipeline assets
├──── *_types.py           # Pydantic model for specifying expected LLM output structure
├──── prompts.py           # Prompt template functions
├──── templates/           # Prompt templates
├──── examples/            # Examples to tailor prompt templates
├── lib/                   # Reusable functionality across pipelines
├──── datagen/             # Bootstrapping synthetic data for LLM fine-tuning
├──── finetune/            # LLM fine-tuning
├──── infer/               # LLM inference
├────── demo/llamadeploy   # Deploy a Llama model on AWS SageMaker AI
├────── demo/deploy        # Deploy LiteLLM proxy for SageMaker AI Llama models
├────── local/deploy       # Deploy model weight distribution infrastructure for llamaserve
├────── local/llamaserve   # Serve llama models locally
├──── utils/               # Reusable functions
└── README.md              # Project overview and documentation
```

<img src="_assets/repo.png" alt="isolated" width="500"/>