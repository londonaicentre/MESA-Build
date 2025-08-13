# Generate synthetic data using Claude 4 Opus/Sonnet on AWS Bedrock

## Arrange access to the required model on AWS Bedrock
Obtain an API access key from an account manager and save as the `BEDROCK_API_KEY` env variable.
Set the AWS region appropriately (in the `claudedatagen.py` script).
Enable model on the AWS Bedrock interface, ask an account manager if not available already. Then set the appropriate model in the `BEDROCK_MODEL` variable.

## Generate synthetic data samples
Using a fresh venv and install the required packages (`pip install -r requirements.txt`).
Run the `claudedatagen.py` script after setting the right variables. This will generate 1500 samples in the model's subfolder under `samples`.