import boto3
import os
import random
import time

from botocore.exceptions import ClientError
from litellm import completion, RateLimitError

def upload_file(region_name, file_name, bucket, object_name=None, path=None):
    if object_name is None:
        object_name = os.path.basename(file_name)
    try:
        boto3.client("s3", region_name=region_name).upload_file(
            file_name, 
            bucket, 
            path + "/" + object_name if path else object_name
        )
    except ClientError as e:
        print(e)
        return False
    return True

def bedrock_completion(model_name, system_prompt, user_prompt, bedrock_api_key):
    max_retries = 5
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
            )
        except RateLimitError:
            if attempt == max_retries:
                raise
            # https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
            delay = random.uniform(0, min(60, 2**attempt))
            print(
                "hit rate limit, waiting "
                + str(round(delay, 2))
                + " seconds (retry "
                + str(attempt + 1)
                + ")"
            )
            time.sleep(delay)