import os

from utils import load_config
from aws import bedrock_completion


def run_bootstrap(
    system_prompt,
    user_prompt_function,
    instruction,
    model_name,
    bedrock_api_key,
):
    config = load_config()
    os.environ["AWS_REGION_NAME"] = config[model_name]["region"]
    message = bedrock_completion(
        config[model_name]["model"],
        system_prompt,
        user_prompt_function(instruction),
        bedrock_api_key,
    )
    with open("bootstrap.csv", "w", newline="") as file:
        file.write(message.choices[0].message.content)
