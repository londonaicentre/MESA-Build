import os
import re
from re import Match
import json
from datetime import datetime
from typing import Any, Callable, cast

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import litellm
from pydantic import BaseModel
from litellm import Choices, ModelResponse

from datagen.config import Config
from utils.aws import upload_file, bedrock_completion

litellm.suppress_debug_info = (
    True  # suppress unhelpful library output on rate limit error
)


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract JSON from Claude's response

    Args:
        response (str): Claude's response

    Returns:
        JSON if parse successful, None otherwise

    """
    # try parse response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # try to find JSON in a code block
        json_match: Match[str] | None = re.search(
            r"```json\s*(.*?)\s*```", response, re.DOTALL
        )
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # If fails, try to find any JSON-like structure
        json_match = re.search(r"{[\s\S]*}", response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                raise Exception("Could not parse JSON from response")

        print("No valid JSON found in response")
        return None


def validate_with_pydantic(
    schema: type[BaseModel], output_data: dict[str, Any]
) -> tuple[bool, BaseModel | None]:
    """Validate the output against the Pydantic schema

    Args:
        schema (type[BaseModel]): Schema class to use for validation
        output_data (dict): JSON data to validate

    Returns:
        True, Parsed JSON if validation successful, False, None otherwise.

    """
    try:
        validated_report: BaseModel = schema(**output_data)
        return True, validated_report
    except Exception as e:
        print(f"Pydantic validation error: {e}")
        return False, None


def process_bootstrap_rows(
    system_prompt: str,
    user_prompt_function: Callable[[dict[str, Any]], str],
    model_name: str,
    bootstrap_file: str,
    output_dir: str,
    bedrock_api_key: str,
    schema: type[BaseModel],
    sample_size: int = 10,
) -> None:
    """Process rows from the specified bootstrap file and
        generate the requested number of samples

    Args:
        system_prompt (str): System prompt for sample generation
        user_prompt_function (Callable): User prompt generation
            function to which bootstrap data is passed
        model_name (str): Name of model to use on AWS Bedrock
        bootstrap_file (str): Path to file with sample configuration
        output_dir (str): Path to output folder
        bedrock_api_key (str): API key to access AWS Bedrock
        schema (type[BaseModel]): Schema class to use for validation
        sample_size (int, optional): Number of samples to generate.
            Defaults to 10.

    """
    df: pd.DataFrame = pd.read_csv(bootstrap_file)

    # Process rows
    os.makedirs(output_dir, exist_ok=True)
    samples_exist: int = len(os.listdir(output_dir))
    successful_generations: int = 0
    failed_generations: int = 0
    if samples_exist > sample_size:
        print(
            f"Requested number of samples have already been generated in {output_dir}."
        )
        exit()
    max_samples: int = len(df.index)
    if sample_size > max_samples:
        print(
            f"Requested number of samples is more than number of templates for generation. \
                Will create {max_samples} samples instead of {sample_size}"
        )
        sample_size = max_samples
    for idx, row in df.iterrows():
        id: int = int(cast(int, idx))

        # Skip rows for which samples have been generated
        if id < samples_exist:
            continue

        # Stop generating samples when requested amount is reached
        if id == sample_size:
            print(f"Generated the requested number of samples, {sample_size}.")
            break
        if generate_sample(
            system_prompt,
            user_prompt_function,
            model_name,
            df,
            id,
            bedrock_api_key,
            schema,
        ):
            successful_generations += 1
        else:
            failed_generations += 1
    print(
        f"Processing complete: {successful_generations} successful, {failed_generations} failed"
    )


def find_missing_idx(folder_name: str, sample_size: int) -> list[int]:
    """For a given folder and expected number of samples,
        identifies indices for which no sample was generated

    Args:
        folder_name (str): Path to the output folder
        sample_size (int): Expected number of samples

    Returns:
        list[int]: List of indices without a sample generated

    """
    all_files: list[str] = os.listdir(folder_name)
    filenames: list[str] = [file.strip(".json").strip("sample") for file in all_files]
    missing_idx: list[int] = []
    for idx in range(sample_size):
        if f"{idx + 1:04d}" not in filenames:
            # print(f"Sample missing for index {idx+1}")
            missing_idx.append(idx)

    return missing_idx


def backfill(
    system_prompt: str,
    user_prompt_function: Callable[[dict[str, Any]], str],
    model_name: str,
    bootstrap_file: str,
    idx_list: list[int],
    bedrock_api_key: str,
    schema: type[BaseModel],
) -> None:
    """Generate samples for the missing indices

    Args:
        system_prompt (str): System prompt for sample generation
        user_prompt_function (Callable): User prompt generation
            function to which bootstrap data is passed
        model_name (str): Name of model to use on AWS Bedrock
        bootstrap_file (str): Path to bootstrap file
        idx_list (list[int]): List of indices for a sample to be generated
        bedrock_api_key (str): API key to access AWS Bedrock
        schema (type[BaseModel]): Schema class to use for validation

    """
    df: pd.DataFrame = pd.read_csv(bootstrap_file)
    successful_generations: int = 0
    failed_generations: int = 0
    for idx in idx_list:
        print(f"Processing row {idx + 1}")
        if generate_sample(
            system_prompt,
            user_prompt_function,
            model_name,
            df,
            idx,
            bedrock_api_key,
            schema,
        ):
            successful_generations += 1
        else:
            failed_generations += 1
    print(
        f"Processing complete: {successful_generations} successful, {failed_generations} failed"
    )


def generate_sample(
    system_prompt: str,
    user_prompt_function: Callable[[dict[str, Any]], str],
    model_name: str,
    df: pd.DataFrame,
    idx: int,
    bedrock_api_key: str,
    schema: type[BaseModel],
) -> bool:
    """Generate synthetic patient reports.

    Args:
        system_prompt (str): System prompt for sample generation
        user_prompt_function (Callable): User prompt generation
            function to which bootstrap data is passed
        model_name (str): Name of model to use on AWS Bedrock
        df (pandas.DataFrame): template for sample report generation
        idx (int): Index for row to be processed from template
        bedrock_api_key (str): API key to access AWS Bedrock
        schema (type[BaseModel]): Schema class to use for validation

    Returns:
        bool: Whether sample generation is successful

    """
    row: pd.Series = df.iloc[idx]
    try:
        user_prompt: str = user_prompt_function(row.to_dict())
        message: ModelResponse | None = bedrock_completion(
            model_name, system_prompt, user_prompt, bedrock_api_key
        )
        if message is None:
            return False
        json_output: dict[str, Any] | None = extract_json_from_response(
            str(cast(Choices, message.choices[0]).message.content)
        )
        if json_output is not None:
            if (
                not isinstance(json_output, dict)
                or "content" not in json_output
                or "output" not in json_output
            ):
                print(f"Invalid schema format in output for row {idx + 1}")
                return False

            # validate against schema
            is_valid: bool
            validated_output: BaseModel | None
            is_valid, validated_output = validate_with_pydantic(
                schema, json_output["output"]
            )
            if is_valid and validated_output is not None:
                # Convert Pydantic model to dict for JSON serialization
                json_output["output"] = validated_output.model_dump()
                output_filename: str = os.path.join(
                    "samples_sonnet4/", f"sample{idx + 1:04d}.json"
                )
                try:
                    with open(output_filename, "w", encoding="utf-8") as f:
                        json.dump(json_output, f, indent=4, ensure_ascii=False)
                    print(f"Successfully saved output to {output_filename}")
                    return True
                except Exception as e:
                    print(f"Error saving JSON to file: {e}")
                    return False
            else:
                print(f"Pydantic validation failed for row {idx + 1}")

                # for debugging later
                debug_filename: str = os.path.join(
                    "samples_sonnet4/",
                    f"invalid_sample{idx + 1:04d}.json",
                )
                with open(debug_filename, "w", encoding="utf-8") as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)
                return False
        else:
            print(f"Skipping file save for row {idx + 1} due to JSON parsing failure")
            print(
                "Claude response:",
                str(cast(Choices, message.choices[0]).message.content),
            )
            return False
    except Exception as e:
        print(f"Error processing row {idx + 1}: {e}")
        return False


def generate_batch(
    system_prompt: str,
    user_prompt_function: Callable[[dict[str, Any]], str],
    bootstrap_file: str,
    sample_size: int,
) -> str:
    """Generate batch request file for Anthropic model

    Args:
        system_prompt (str): System prompt for sample generation
        user_prompt_function (Callable): User prompt generation
            function to which bootstrap data is passed
        bootstrap_file (str): Path to bootstrap file
        sample_size (int): Number of samples to be generated

    Returns:
        str: The batch request file

    """
    fn: str = "anthropic_batch_job.jsonl"
    df: pd.DataFrame = pd.read_csv(bootstrap_file)
    max_samples: int = len(df.index)
    if sample_size > max_samples:
        print(
            f"Requested number of samples is more than number of templates for generation. \
                Will create {max_samples} samples instead of {sample_size}"
        )
        sample_size = max_samples
    with open(fn, "w") as outfile:
        for idx, row in df.iterrows():
            # Stop generating samples when requested amount is reached
            if idx == sample_size:
                print(f"Generated the requested number of samples, {sample_size}.")
                break
            user_prompt: str = user_prompt_function(row.to_dict())
            record: dict[str, Any] = {
                "recordId": str(idx),
                "modelInput": {
                    "anthropic_version": "bedrock-2023-05-31",
                    "system": system_prompt,
                    "max_tokens": 4000,
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
            print(json.dumps(record), file=outfile)
    return fn


def start_batch_inference(
    region_name: str,
    job_id: str,
    model_id: str,
    role_arn: str,
    bucket: str,
    batch_file: str,
) -> bool:
    """Start AWS Bedrock batch inference

    Args:
        region_name (str): The region in which to run the batch
        job_id (str): The id to give to the batch job
        model_id (str): The id of the model to target for the job
        role_arn (str): The ARN of an IAM role with permissions to
            access S3 for batch specification and access
            cross-region models
        bucket (str): The name of the bucket in which the batch
            specification exists
        batch_file (str): The name of the batch specification file

    Returns:
        bool: Whether the batch inference run started successfully

    """
    try:
        boto3.client("bedrock", region_name=region_name).create_model_invocation_job(
            jobName="schemallama-" + job_id.replace("/", "-"),
            modelId=model_id,
            roleArn=role_arn,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": "s3://" + bucket + "/" + job_id + "/input/" + batch_file
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


def run_batch_inference(
    system_prompt: str,
    user_prompt_function: Callable[[dict[str, Any]], str],
    model_name: str,
    template: str,
    sample_size: int,
    bucket: str,
    bedrock_execution_role: str,
) -> None:
    """Generate samples via batch inference

    Args:
        system_prompt (str): System prompt for sample generation
        user_prompt_function (Callable): User prompt generation
            function to which bootstrap data is passed
        model_name (str): Name of model to use on AWS Bedrock
        template (str): Path to bootstrap (template) file
        sample_size (int): Number of samples to be generated
        bucket (str): The name of the bucket in which the batch
            specification exists
        bedrock_execution_role (str): The ARN of an IAM role with
            permissions to access S3 for batch specification and
            access cross-region models

    """
    config: Config = Config()

    # Process all samples from bootstrap file in batch mode
    # Create batch instruction JSONL file
    generate_batch(system_prompt, user_prompt_function, template, sample_size)
    job_id: str = "datagen/" + datetime.now().strftime("%Y-%m-%d-%H%M")

    # Upload to S3 bucket
    upload_file(
        config.models[model_name].region,
        config.models[model_name].batch_file,
        bucket,
        config.models[model_name].batch_file,
        job_id + "/input",
    )

    # Generate samples in batch mode
    start_batch_inference(
        config.models[model_name].region,
        job_id,
        config.models[model_name].model,
        bedrock_execution_role,
        bucket,
        config.models[model_name].batch_file,
    )


def run_backfill(
    system_prompt: str,
    user_prompt_function: Callable[[dict[str, Any]], str],
    model_name: str,
    template: str,
    sample_size: int,
    bedrock_api_key: str,
    schema: type[BaseModel],
) -> None:
    """Backfill missing samples

    Args:
        system_prompt (str): System prompt for sample generation
        user_prompt_function (Callable): User prompt generation
            function to which bootstrap data is passed
        model_name (str): Name of model to use on AWS Bedrock
        template (str): Path to bootstrap (template) file
        sample_size (int): Number of samples to be generated
        bedrock_api_key (str): API key to access AWS Bedrock
        schema (type[BaseModel]): Schema class to use for validation

    """
    config: Config = Config()
    os.environ["AWS_REGION_NAME"] = config.models[model_name].region
    folder_name: str = f"samples_{model_name}/"

    # Generate samples for missed indices in the bootstrap file specified
    missing_idx: list[int] = find_missing_idx(folder_name, sample_size)
    print(f"There are {len(missing_idx)} samples missing")
    backfill(
        system_prompt,
        user_prompt_function,
        config.models[model_name].model,
        template,
        missing_idx,
        bedrock_api_key,
        schema,
    )


def run_sample_generation(
    system_prompt: str,
    user_prompt_function: Callable[[dict[str, Any]], str],
    model_name: str,
    template: str,
    sample_size: int,
    bedrock_api_key: str,
    schema: type[BaseModel],
) -> None:
    """Generate samples via individual AWS Bedrock inference calls

    Args:
        system_prompt (str): System prompt for sample generation
        user_prompt_function (Callable): User prompt generation
            function to which bootstrap data is passed
        model_name (str): Name of model to use on AWS Bedrock
        template (str): Path to bootstrap (template) file
        sample_size (int): Number of samples to be generated
        bedrock_api_key (str): API key to access AWS Bedrock
        schema (type[BaseModel]): Schema class to use for validation

    """
    config: Config = Config()
    os.environ["AWS_REGION_NAME"] = config.models[model_name].region
    folder_name = f"samples_{model_name}/"

    # Generate samples from bootstrap file
    process_bootstrap_rows(
        system_prompt,
        user_prompt_function,
        config.models[model_name].model,
        template,
        folder_name,
        bedrock_api_key,
        schema,
        sample_size,
    )


def run_bootstrap_file_generation(
    system_prompt: str,
    user_prompt_function: Callable[[str], str],
    instruction: str,
    model_name: str,
    bedrock_api_key: str,
) -> None:
    """Generate bootstrap file to vary samples

    Args:
        system_prompt (str): System prompt for bootstrap file generation
        user_prompt_function (Callable): User prompt generation
            function to which instruction is passed
        instruction (str): Instruction to tailor bootstrap file to
            specific area
        model_name (str): Name of model to use on AWS Bedrock
        bedrock_api_key (str): API key to access AWS Bedrock

    """
    config: Config = Config()
    os.environ["AWS_REGION_NAME"] = config.models[model_name].region
    message: ModelResponse | None = bedrock_completion(
        config.models[model_name].model,
        system_prompt,
        user_prompt_function(instruction),
        bedrock_api_key,
    )
    if message is not None:
        with open("bootstrap.csv", "w", newline="") as file:
            file.write(str(cast(Choices, message.choices[0]).message.content))
