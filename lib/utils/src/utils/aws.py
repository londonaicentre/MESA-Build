import os
import random
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError
from litellm import RateLimitError, ModelResponse
from utils.llm import LLM


class AWS:
    @staticmethod
    def upload_file(
        region_name: str,
        file_name: str,
        bucket: str,
        object_name: str | None = None,
        path: str | None = None,
    ) -> bool:
        """Upload a file to S3

        Args:
            region_name (str): The region in which the bucket exists
            file_name (str): The name of the local file to upload
            bucket (str): The name of the target bucket
            object_name (str, optional): the name of the uploaded object.
                If absent, file_name is used.
            path (str, optional): the path to the uploaded object. If absent,
                file_name is used.

        Returns:
            bool: Whether the upload was successful

        """
        if object_name is None:
            object_name = os.path.basename(file_name)
        try:
            boto3.client("s3", region_name=region_name).upload_file(
                file_name, bucket, path + "/" + object_name if path else object_name
            )
        except ClientError as e:
            print(e)
            return False
        return True

    @staticmethod
    def bedrock_completion(
        model_name: str,
        system_prompt: str | None,
        user_prompt: str,
        bedrock_api_key: str,
        max_tokens: int = 8192,
        temperature: float = 0.001,
    ) -> ModelResponse | None:
        """Use a Bedrock LLM for inference. Uses backoff and jitter on rate limit.

        Args:
            model_name (str): The name of the LLM
            system_prompt (str): The system prompt to use
            user_prompt (str): The user prompt to use
            bedrock_api_key (str): API key to access AWS Bedrock
            max_tokens (int): Maximum output tokens. Defaults to 8192.
            temperature (float): Model randomness. Defaults to 0.001.

        Returns:
            ModelResponse: The model's prediction (LiteLLM wrapper object)

        """
        max_retries: int = 5
        for attempt in range(max_retries + 1):
            try:
                return LLM.completion(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    api_key=bedrock_api_key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    aws_region_name="us-east-1",
                )
            except RateLimitError:
                if attempt == max_retries:
                    raise
                # https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
                delay: float = random.uniform(0, min(60, 2**attempt))
                print(
                    "hit rate limit, waiting "
                    + str(round(delay, 2))
                    + " seconds (retry "
                    + str(attempt + 1)
                    + ")"
                )
                time.sleep(delay)
        return None

    @staticmethod
    def create_anthropic_bedrock_batch_entry(
        id: str, system_prompt: str | None, user_prompt: str, max_tokens: int = 4000
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "recordId": id,
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt,
                            }
                        ],
                    },
                ],
            },
        }
        if system_prompt is not None:
            record["modelInput"]["system"] = system_prompt
        return record

    @staticmethod
    def create_model_invocation_job(
        job_id: str,
        model_id: str,
        batch_file: str,
        bucket: str,
        bedrock_execution_role: str,
        model_region: str,
    ) -> bool:
        """Create a model invocation job (batch inference run)
            on AWS Bedrock

        Args:
            job_id (str): The id to give to the batch job
            model_id (str): The Bedrock id of the model to use for
                inference in the batch job
            batch_file (str): The name of the local file
                containing the batch specification
            bucket (str): The name of the bucket in which the batch
                specification exists
            bedrock_execution_role (str): The ARN of an IAM role with
                permissions to access S3 for batch specification and
                access cross-region models
            model_region (str): The region in which to run the job

        Returns:
            bool: Whether the batch inference run started successfully

        """
        try:
            boto3.client(
                "bedrock", region_name=model_region
            ).create_model_invocation_job(
                jobName="schemallama-" + job_id.replace("/", "-"),
                modelId=model_id,
                roleArn=bedrock_execution_role,
                inputDataConfig={
                    "s3InputDataConfig": {
                        "s3Uri": "s3://"
                        + bucket
                        + "/"
                        + job_id
                        + "/input/"
                        + batch_file
                    }
                },
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": "s3://" + bucket + "/" + job_id + "/output/"
                    }
                },
            )
        except ClientError as e:
            print(e)
            return False
        return True

    @staticmethod
    def run_batch_inference(
        job_id: str,
        model_id: str,
        batch_file: str,
        bucket: str,
        bedrock_execution_role: str,
        model_region: str,
    ) -> None:
        """Generate samples via batch inference

        Args:
            job_id (str): The id to give to the batch job
            model_id (str): The Bedrock id of the model to use for
                inference in the batch job
            batch_file (str): The name of the local file
                containing the batch specification
            bucket (str): The name of the bucket to which the batch
                specification should be uploaded
            bedrock_execution_role (str): The ARN of an IAM role with
                permissions to access S3 for batch specification and
                access cross-region models
            model_region (str): The region in which to run the job

        """
        # Upload to S3 bucket
        AWS.upload_file(
            model_region,
            batch_file,
            bucket,
            batch_file,
            job_id + "/input",
        )

        # Generate samples in batch mode
        AWS.create_model_invocation_job(
            job_id, model_id, batch_file, bucket, bedrock_execution_role, model_region
        )
