# MESA: Training Data Generation

Training data generation for MESA models.

See [examples](examples/README.md).

## Quickstart (AWS batch generation)

1. Follow the instructions [here](https://github.com/londonaicentre/sde_aic_internal_docs/blob/main/nlp/using_amazon_sagemaker.md#sso-authentication-for-use-of-the-aws-cli) to set up SSO authentication for use of the AWS CLI.

2. Obtain information on a Bedrock Execution IAM Role with S3 and model access and information on the name of an S3 bucket to upload a batch specification to.

3. Download documents from batch

    ```python
    from datagen import BedrockBatchGenerator
    gen = BedrockBatchGenerator(
        system_prompt =
        user_prompt_function =
        schema =
        schema_name =
        model_name =
        document_batches =
    )
    ```

4. Start batch generation

    ```python
    gen.generate_via_batch(
        sample_size =
        bedrock_execution_role =
        bucket = 
    )
    ```

5. Download and parse batch outputs

    ```python
    gen.extract_batch_output(
        bucket = 
        file_name =
    )
    ```

6. Upload formatted document:schema pairs as training data

    ```python
    from datagen import TrainingDataUploader
    s3_uri = TrainingDataUploader.upload(
        schema =
        schema_name =
        system_prompt =
        short_description =
        long_description =
        input_folder =
    )
    ```
