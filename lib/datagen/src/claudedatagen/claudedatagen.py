import litellm

litellm.suppress_debug_info = (
    True  # suppress unhelpful library output on rate limit error
)
import os
import re
import json
import boto3
import pandas as pd
from datetime import datetime
from botocore.exceptions import ClientError
from typing import Callable

from aws import upload_file, bedrock_completion
from utils import load_config


def extract_json_from_response(response):
    """
    Extract JSON from Claude's response
    """
    # claude may return a text block or a list...
    if hasattr(response, "text"):
        response = response.text
    elif isinstance(response, list):
        response = response[0].text

    # try parse response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # try to find JSON in a code block
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
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


def validate_with_pydantic(schema, output_data):
    """
    Validate the output against the Pydantic schema
    """
    try:
        validated_report = schema(**output_data)
        return True, validated_report
    except Exception as e:
        print(f"Pydantic validation error: {e}")
        return False, None


def process_bootstrap_rows(
    system_prompt: str,
    user_prompt_function: Callable[[dict], str],
    model_name: str,
    bootstrap_file,
    output_dir: str,
    bedrock_api_key: str,
    schema,
    sample_size: int = 10,
) -> None:
    """
    Process rows from the specified bootstrap file and generate the requested number of samples.

    Args:
        system_prompt (str): _description_
        model_name (str): Name of model to use on AWS Bedrock.
        bootstrap_file (str): Path to file with sample configuration.
        output_dir (str): Path to output folder.
        sample_size (int): Number of samples to generate.
        examples_dir (str): Path to example files.
    """
    df = pd.read_csv(bootstrap_file)

    ## PROCESS ROWS

    os.makedirs(output_dir, exist_ok=True)
    samples_exist = len(os.listdir(output_dir))

    successful_generations = 0
    failed_generations = 0

    if samples_exist > sample_size:
        print(
            f"Requested number of samples have already been generated in {output_dir}."
        )
        exit()

    max_samples = len(df.index)
    if sample_size > max_samples:
        print(
            f"Requested number of samples is more than number of templates for generation. \
                Will create {max_samples} samples instead of {sample_size}"
        )
        sample_size = max_samples

    for idx, row in df.iterrows():
        # Skip rows for which samples have been generated
        if idx < samples_exist:
            continue

        # Stop generating samples when requested amount is reached
        if idx == sample_size:
            print(f"Generated the requested number of samples, {sample_size}.")
            break

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


def find_missing_idx(folder_name, sample_size) -> list[int]:
    """For a given folder and expected number of samples, identifies indices for which no sample was generated.

    Args:
        folder_name (str): Path to the output folder.
        sample_size (int): Expected number of samples.

    Returns:
        list[int]: List of indices without a sample generated.
    """
    all_files = os.listdir(folder_name)
    filenames = [file.strip(".json").strip("sample") for file in all_files]

    missing_idx = []
    for idx in range(sample_size):
        if f"{idx + 1:04d}" not in filenames:
            # print(f"Sample missing for index {idx+1}")
            missing_idx.append(idx)

    return missing_idx


def backfill(
    system_prompt,
    user_prompt_function,
    model_name,
    bootstrap_file,
    idx_list,
    bedrock_api_key,
    schema,
) -> None:
    """Generate samples for the missing indices.

    Args:
        system_prompt (str): _description_
        model_name (str): Name of model to use on AWS Bedrock.
        bootstrap_file (str): Path to bootstrap file.
        idx_list (list[int]): List of indices for a sample to be generated.
    """

    df = pd.read_csv(bootstrap_file)

    successful_generations = 0
    failed_generations = 0

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
    system_prompt, user_prompt_function, model_name, df, idx, bedrock_api_key, schema
) -> bool:
    """Generate synthetic patient reports.

    Args:
        system_prompt (str): _description_
        model_name (str): Name of model to use on AWS Bedrock.
        df (pandas.DataFrame): template for sample report generation.
        idx (int): Index for row to be processed from template.
    """

    row = df.iloc[idx]

    try:
        user_prompt = user_prompt_function(row)
        message = bedrock_completion(
            model_name, system_prompt, user_prompt, bedrock_api_key
        )
        json_output = extract_json_from_response(message.choices[0].message.content)

        if json_output is not None:
            if (
                not isinstance(json_output, dict)
                or "content" not in json_output
                or "output" not in json_output
            ):
                print(f"Invalid schema format in output for row {idx + 1}")
                return False

            # validate against schema
            is_valid, validated_output = validate_with_pydantic(
                schema, json_output["output"]
            )

            if is_valid:
                # Convert Pydantic model to dict for JSON serialization
                json_output["output"] = validated_output.model_dump()

                output_filename = os.path.join(
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
                debug_filename = os.path.join(
                    "samples_sonnet4/",
                    f"invalid_sample{idx + 1:04d}.json",
                )
                with open(debug_filename, "w", encoding="utf-8") as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)
                return False
        else:
            print(f"Skipping file save for row {idx + 1} due to JSON parsing failure")
            print("Claude response:", message.content)
            return False

    except Exception as e:
        print(f"Error processing row {idx + 1}: {e}")
        return False


def generate_batch(
    system_prompt, user_prompt_function, bootstrap_file, sample_size
):
    """Generate batch request file for Anthropic model.

    Args:
        system_prompt (str): _description_
        bootstrap_file (str): Path to bootstrap file.
        sample_size (int): Number of samples to be generated.
    """

    fn = "anthropic_batch_job.jsonl"

    df = pd.read_csv(bootstrap_file)
    max_samples = len(df.index)

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

            user_prompt = user_prompt_function(row)

            record = {
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


def start_batch_inference(region_name, job_id, model_id, role_arn, bucket, batch_file):
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
    system_prompt,
    user_prompt_function,
    model_name,
    template,
    sample_size,
    bucket,
    bedrock_execution_role,
):
    config = load_config()
    # Process all samples from bootstrap file in batch mode
    # Create batch instruction JSONL file
    batch_jsonl = generate_batch(
        system_prompt, user_prompt_function, template, sample_size
    )
    job_id = "datagen/" + datetime.now().strftime("%Y-%m-%d-%H%M")
    # Upload to S3 bucket
    upload_file(
        config[model_name]["region"],
        config[model_name]["batch_file"],
        bucket,
        config[model_name]["batch_file"],
        job_id + "/input",
    )
    # Generate samples in batch mode
    start_batch_inference(
        config[model_name]["region"],
        job_id,
        config[model_name]["model"],
        bedrock_execution_role,
        bucket,
        config[model_name]["batch_file"],
    )


def run_backfill(
    system_prompt,
    user_prompt_function,
    model_name,
    template,
    sample_size,
    bedrock_api_key,
    schema,
):
    config = load_config()
    os.environ["AWS_REGION_NAME"] = config[model_name]["region"]
    folder_name = f"samples_{model_name}/"
    # Generate samples for missed indices in the bootstrap file specified
    missing_idx = find_missing_idx(folder_name, sample_size)
    print(f"There are {len(missing_idx)} samples missing")
    backfill(
        system_prompt,
        user_prompt_function,
        config[model_name]["model"],
        template,
        missing_idx,
        bedrock_api_key,
        schema,
    )


def run_sample_generation(
    system_prompt,
    user_prompt_function,
    model_name,
    template,
    sample_size,
    bedrock_api_key,
    schema,
):
    config = load_config()
    os.environ["AWS_REGION_NAME"] = config[model_name]["region"]
    folder_name = f"samples_{model_name}/"
    # Generate samples from bootstrap file
    process_bootstrap_rows(
        system_prompt,
        user_prompt_function,
        config[model_name]["model"],
        template,
        folder_name,
        bedrock_api_key,
        schema,
        sample_size,
    )

def run_bootstrap_file_generation(
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
