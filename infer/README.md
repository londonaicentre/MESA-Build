# Deploy a Llama model on AWS SageMaker AI (demo)

## Arrange access to SageMaker
Follow the instructions [here](https://github.com/londonaicentre/sde_aic_internal_docs/blob/main/nlp/using_amazon_sagemaker.md#sso-authentication-for-use-of-the-aws-cli) to set up SSO authentication for use of the AWS CLI.

## Deploy a Llama model
Using a venv with the required packages installed (`pip install -r requirements.txt`) and after setting the right env variables, run `llamadeploy.py`.
Some command line arguments are optional, set what you need to change:
`-d, --demo`: Whether to demo inference by deploying to a remote AWS SageMaker AI Endpoint. Defaults to `False`.
`-p, --path`: Path within S3 bucket to the zipped weights of the model to deploy. Required.
`[command]`: The action to perform: `up` (deploy) or `down` (delete). Required.
