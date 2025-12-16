import os
from pathlib import Path
import random
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError
from litellm import RateLimitError, ModelResponse
from pydantic import BaseModel

from utils.llm import LLM, Message, TextContent


class ModelInput(BaseModel):
    anthropic_version: str = "bedrock-2023-05-31"
    system: str | None
    max_tokens: int
    messages: list[Message]


class AnthropicBedrockBatchEntry(BaseModel):
    recordId: str
    modelInput: ModelInput


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
    def download_file(
        region_name: str,
        bucket: str,
        file_name: str,
        object_name: str | None = None,
        path: str | None = None,
    ) -> bool:
        """Download a file from S3

        Args:
            region_name (str): The region in which the bucket exists
            bucket (str): The name of the target bucket
            file_name (str): The name to use for the downloaded file
            object_name (str, optional): the name of the object to download.
                If absent, file_name is used.
            path (str, optional): the path to the target object. If absent,
                file_name is used.

        Returns:
            bool: Whether the upload was successful

        """
        if object_name is None:
            object_name = os.path.basename(file_name)
        try:
            boto3.client("s3", region_name=region_name).download_file(
                bucket, path + "/" + object_name if path else object_name, file_name
            )
        except ClientError as e:
            print(e)
            return False
        return True

    @staticmethod
    def download_file_with_wildcard(
        region_name: str,
        bucket: str,
        file_name: str,
        object_name: str,
        path: str,
    ) -> bool:
        """Download a file from S3 with a path that contains a wildcard

        Args:
            region_name (str): The region in which the bucket exists
            bucket (str): The name of the target bucket
            file_name (str): The name to use for the downloaded file
            object_name (str, optional): the name of the object to download.
            path (str): the path to the target object. Can contain
                a wildcard.

        Returns:
            bool: Whether the upload was successful

        """
        prefix: str
        suffix: str
        prefix, suffix = (path + "/" + object_name).split("*/", 1)
        for page in (
            boto3.client("s3", region_name=region_name)
            .get_paginator("list_objects_v2")
            .paginate(Bucket=bucket, Prefix=prefix)
        ):
            for object in page.get("Contents", []):
                key: str = object["Key"]
                if key.endswith(suffix):
                    return AWS.download_file(
                        region_name,
                        bucket,
                        file_name,
                        object_name,
                        str(Path(key).parent),
                    )
        return False

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
                    aws_region_name="eu-west-2",
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
        id: str, system_prompt: str | None, user_prompt: str, max_tokens: int = 8192
    ) -> dict[str, Any]:
        """Create an entry for a Bedrock batch execution file targeting
            Anthropic models.

        Args:
            id (str): Unique id of the entry in the resulting file
            system_prompt (str, optional): The system prompt to use
                during batch inference
            user_prompt (str): The user prompt to use during batch inference
            max_tokens (int, optional): The maximum number of output tokens

        Returns:
            dict: The batch entry object as a dictionary

        """
        return AnthropicBedrockBatchEntry(
            recordId=id,
            modelInput=ModelInput(
                max_tokens=max_tokens,
                messages=[
                    Message(
                        role="user",
                        content=[TextContent(type="text", text=user_prompt)],
                    )
                ],
                system=system_prompt,
            ),
        ).model_dump(exclude_none=True)

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
