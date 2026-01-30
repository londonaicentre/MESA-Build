# MESA: Training Data Generation

Training data generation for MESA models.

## Getting started

### AWS

- Obtain a Bedrock API key from an account manager.

- (Batch inference only) Obtain information on a Bedrock Execution IAM Role with S3 and model access.

- (Batch inference only) Obtain information on the name of an S3 bucket to upload a batch specification to.

## Usage

1. Download documents from batch

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

2. Start batch generation

    ```python
    gen.generate_via_batch(
        sample_size =
        bucket = 
        bedrock_execution_role =
    )
    ```

3. Download and parse batch outputs

    ```python
    gen.extract_batch_output(
        bucket = 
        file_name =
    )
    ```

4. Upload formatted document:schema pairs as training data

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
