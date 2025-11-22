# SchemaLlama: Fine-tune

Fine-tune a Llama model on AWS Sagemaker.

## Getting started

### AWS

- Follow the instructions [here](https://github.com/londonaicentre/sde_aic_internal_docs/blob/main/nlp/using_amazon_sagemaker.md#sso-authentication-for-use-of-the-aws-cli) to set up SSO authentication for use of the AWS CLI.

## Usage

1. Start a fine-tuning run:

```python
from finetune.llama import run_finetune
run_finetune(
    <System prompt>,
    <Samples file>, 
    <Bucket>, 
    <Sagemaker execution role>, 
    <Instance flavour>
)
```
