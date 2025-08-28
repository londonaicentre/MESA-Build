# Generate synthetic data using Claude 4 Opus/Sonnet on AWS Bedrock

## Arrange access to the required model on AWS Bedrock
Obtain an API access key from an account manager and save as the `BEDROCK_API_KEY` env variable.
Set the AWS region appropriately (in the `claudedatagen.py` script).
Enable model on the AWS Bedrock interface, ask an account manager if not available already. Then set the appropriate model in the `BEDROCK_MODEL` variable.

## Generate synthetic data samples
Using a venv with the required packages installed (`pip install -r requirements.txt`),
Run the `claudedatagen.py` script after setting the right env variables. The required command line arguments are:
`model_name`:  Name of model to use, eg 'sonnet4' or 'opus4', must have corresponding configuration in the `config.json` file.
`sample_size`: Required number of samples to be generated.
`--backfill`: OPTIONAL argument to generate additional samples and backfill for missed indices.


For example `python claudedatagen.py sonnet4 15` will generate 15 samples in the model's subfolder under `samples`,
while `python claudedatagen.py sonnet4 23 -b True` will generate sample reports for any missed indices and at least 8 additional samples.