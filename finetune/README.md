# Fine-tune a Llama model on AWS Sagemaker

## Arrange access to Sagemaker
Follow the instructions [here](https://github.com/londonaicentre/sde_aic_internal_docs/blob/main/nlp/using_amazon_sagemaker.md#sso-authentication-for-use-of-the-aws-cli) to set up SSO authentication for use of the AWS CLI.

## Start a fine-tuning run
Using a venv with the required packages installed (`pip install -r requirements.txt`) and after setting the right env variables, run `llamafinetune.py`.