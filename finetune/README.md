# Fine-tune a Llama model on AWS Sagemaker

## Arrange access to Sagemaker
Follow the instructions [here](https://github.com/londonaicentre/sde_aic_internal_docs/blob/main/nlp/using_amazon_sagemaker.md#sso-authentication-for-use-of-the-aws-cli) to set up SSO authentication for use of the AWS CLI.

## Start a fine-tuning run
Using a venv with the required packages installed (`pip install -r requirements.txt`) and after setting the right env variables, run `llamafinetune.py`.
All command line arguments are optional, set what you need to change:
`-f, --file`: Name of the AWS Bedrock Anthropic batch inference output file containing sample data to use as input. Defaults to `anthropic_batch_job.jsonl.out`.
`-d, --dry_run`: Whether to simulate calling AWS endpoints. Defaults to `False`.