# MESA: Training Data Generation

Training data generation for MESA models.

See [examples](examples/).

## Quickstart (AWS batch generation)

1. Obtain access to AWS from your account manager and follow the instructions [here](https://docs.commonfate.io/granted/getting-started) to set up SSO authentication for use of the AWS CLI.

2. Obtain information on a Bedrock Execution IAM Role with S3 and model access and information on the name of an S3 bucket to upload a batch specification to.

3. Download documents from batch

    ```python
    from datagen import BedrockBatchGenerator
    gen = BedrockBatchGenerator(
        system_prompt =
        schema =
        schema_name =
        model_name =
        document_batches =
    )
    ```

4. Estimate the upper-bound cost, in USD, of each sample size you may generate, with one call per size

    ```python
    gen.estimate_cost(3000)
    ```

5. Start batch generation

    ```python
    gen.generate_via_batch(
        sample_size =
        bedrock_execution_role =
        bucket = 
    )
    ```

6. Download and parse batch outputs

    ```python
    gen.extract_batch_output(
        bucket = 
        file_name =
    )
    ```

7. Upload formatted document:schema pairs as training data

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

## Quickstart (MESA Runtime batch generation)

1. Follow step 1 of the AWS quickstart, and obtain the name of an S3 bucket to upload a batch specification to, along with the base URL, username and password of the MESA Runtime API.

2. Download documents from batch, giving the `model_name` of a `config.json` entry whose `model` the runtime serves, or omitting it to split samples across every model the runtime serves

    ```python
    from datagen import MESARuntimeBatchGenerator
    gen = MESARuntimeBatchGenerator(
        system_prompt =
        schema =
        schema_name =
        document_batches =
        base_url =
        username =
        password =
    )
    ```

3. Start batch generation

    ```python
    run_id = gen.generate_via_batch(
        sample_size =
        bucket =
    )
    ```

4. Check the run's `status`, `conclusion` and `html_url`

    ```python
    gen.get_batch_run(run_id)
    ```

5. Follow steps 6 and 7 of the AWS quickstart.

## License

londonaicentre-mesa-datagen © 2026 by London AI Centre is licensed under CC BY-NC-ND 4.0.
To view a copy of this license, visit <https://creativecommons.org/licenses/by-nc-nd/4.0/>
