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
