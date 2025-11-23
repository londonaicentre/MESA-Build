import os
import random
import time

import boto3
from botocore.exceptions import ClientError
from litellm import completion, RateLimitError, ModelResponse


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


def bedrock_completion(
    model_name: str, system_prompt: str, user_prompt: str, bedrock_api_key: str
) -> ModelResponse | None:
    """Use a Bedrock LLM for inference. Uses backoff and jitter on rate limit.

    Args:
        model_name (str): The name of the LLM
        system_prompt (str): The system prompt to use
        user_prompt (str): The user prompt to use
        bedrock_api_key (str): API key to access AWS Bedrock

    Returns:
        ModelResponse: The model's prediction (LiteLLM wrapper object)

    """
    max_retries: int = 5
    for attempt in range(max_retries + 1):
        try:
            return completion(
                model=model_name,
                max_tokens=8192,
                temperature=0.001,
                messages=[
                    {"content": system_prompt, "role": "system"},
                    {"content": user_prompt, "role": "user"},
                ],
                api_key=bedrock_api_key,
                stream=False,
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
