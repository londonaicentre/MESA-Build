# MESA: Training Data Generation

Training data generation for MESA models.

## Getting started

### AWS

- Obtain a Bedrock API key from an account manager.

- (Batch inference only) Obtain information on a Bedrock Execution IAM Role with S3 and model access.

- (Batch inference only) Obtain information on the name of an S3 bucket to upload a batch specification to.

- Enable [one of the target models](src/claudedatagen/config/config.json) on the AWS Bedrock interface. Ask an account manager if not available already.

## Usage

1. Generate a bootstrap file:

    ```python
    from datagen.sample_generator import run_bootstrap_file_generation
    run_bootstrap_file_generation(
        <System prompt>,
        <User prompt function>, 
        <Customisation instruction>,
        <Target model name>, 
        <Bedrock API key>
    )
    ```

2. Generate synthetic input data, e.g.:

    ```python
    from datagen.sample_generator import run_sample_generation
    run_sample_generation(
        <System prompt>,
        <User prompt function>,
        <Target model name>,
        <Bootstrap file>,
        <Number of samples>,
        <Bedrock API key>,
        <Schema>
    )
    ```
