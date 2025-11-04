# Generate synthetic data using Claude 4 Opus/Sonnet on AWS Bedrock

## Arrange access to the required model on AWS Bedrock

Obtain an API access key from an account manager and save as the `BEDROCK_API_KEY` env variable.
Set the AWS region appropriately (in the `claudedatagen.py` script).
Enable model on the AWS Bedrock interface, ask an account manager if not available already. Then set the appropriate model in the `BEDROCK_MODEL` variable.

## Generate synthetic data samples

Using a venv with the required packages installed (`pip install -r requirements.txt`),
Run the `claudedatagen.py` script after setting the right env variables. All command line arguments are optional, set what you need to change:

- `-m, --model_name`:  Name of model to use, eg 'sonnet4' or 'opus4', must have corresponding configuration in the `config.json` file. Defaults to `sonnet4`.

- `-s, --sample_size`: Required number of samples to be generated, defaults to `10`.

- `-b, --bootstrap`: Path to the bootstrap file (must be in the same directory or provide absolute path). Defaults to `bootstrap.csv`.

- `-f, --backfill`: Whether to generate additional samples and backfill for missed indices from bootstrap file. Defaults to `False`.

For example `python claudedatagen.py sonnet4 15` will generate 15 samples in the model's subfolder under `samples`,
while `python claudedatagen.py sonnet4 23 -b True` will generate sample reports for any missed indices and at least 8 additional samples.