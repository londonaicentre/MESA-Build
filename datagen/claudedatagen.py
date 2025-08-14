import argparse
import litellm

litellm.suppress_debug_info = (
    True  # suppress unhelpful library output on rate limit error
)
from litellm import completion, RateLimitError
import os
import sys
import re
import time
import json
import random
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from schema.genomicextractmodel import GenomicTestReport


def load_config(configlocation="config.json"):
    try:
        with open(configlocation, "r") as file:
            config = json.load(file)
            print("Successfully loaded config file.")
            return config
    except FileNotFoundError:
        print("Failed to load {configlocation}")
        raise
    except json.JSONDecodeError as json_error:
        print(f"Failed to parse {configlocation} as it has {json_error}")
        raise
    except Exception as e:
        print(f"Failed to load {configlocation} due to {e}")
        raise


def parse_CLI_args():  # -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        args : Namespace
            Namespace of passed command line argument inputs
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_name",
        type=str,
        default="sonnet4",
        help="Name of model to use, eg sonnet4 or opus4",
    )
    parser.add_argument(
        "sample_size",
        type=int,
        default=10,
        help="Number of samples to generate",
    )
    args = parser.parse_args()
    return args


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


def validate_with_pydantic(output_data):
    """
    Validate the output against the Pydantic schema
    """
    try:
        validated_report = GenomicTestReport(**output_data)
        return True, validated_report
    except Exception as e:
        print(f"Pydantic validation error: {e}")
        return False, None


def process_bootstrap_rows(
    model_name: str,
    bootstrap_file,
    output_dir: str,
    sample_size: int = 10,
    examples_dir="examples",
):
    """
    Process rows from bootstrap.csv and generate synthetic genomic reports

    Args:
        model_name (str): name of the model to use
        bootstrap_file: path to file with sample configuration
        output_dir (str): path to output folder
        sample_size (int): number of samples to generate
        examples_dir (str): path to example files
    """
    df = pd.read_csv(bootstrap_file)

    ## CREATE SYSTEM PROMPT
    # schema
    with open("../schema/genomicextractmodel.py", "r") as f:
        schema_content = f.read()

    # examples
    examples_path = Path(examples_dir)
    e1 = ""
    e2 = ""
    try:
        with open(examples_path / "e1.json", "r") as f:
            e1 = f.read()
        with open(examples_path / "e2.json", "r") as f:
            e2 = f.read()
    except FileNotFoundError as e:
        print(f"Warning: Could not load example file: {e}")

    # prompt
    with open("systemprompt.md", "r") as f:
        system_prompt_template = f.read()

    # create full system prompt using replace instead of format to avoid issues with curly braces
    system_prompt = (
        system_prompt_template.replace("{schema_content}", schema_content)
        .replace("{e1}", e1)
        .replace("{e2}", e2)
    )

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
            f"Requested number of samples is more than number of templates for generation. Will create {max_samples} samples instead of {sample_size}"
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

        print(f"Processing row {idx + 1}/{sample_size}")

        try:
            user_prompt = f"""Please generate a genomic laboratory report based on the following test scenario:

            Test Type: {row['test_type']}
            Test Details: {row['test_details']}
            Result Entities: {row['result_entities']}
            Result Description: {row['result_description']}
            Clinical Context: {row['clinical_context']}
            Disease Context: {row['disease_context']}
            Family History: {row['family_history']}
            Test Subject: {row['test_subject']}
            Clinical Implications: {row['clinical_implications']}
            Recommendations: {row['recommendations']}
            Report Style: {row['report_style']}

            Generate a realistic genomic laboratory report incorporating all these details.
            Then extract the information into the structured schema format."""

            max_retries = 5
            for attempt in range(max_retries + 1):
                try:
                    message = completion(
                        model=model_name,
                        max_tokens=8192,
                        temperature=0.001,
                        messages=[
                            {"content": system_prompt, "role": "system"},
                            {"content": user_prompt, "role": "user"},
                        ],
                        api_key=os.environ["BEDROCK_API_KEY"],
                    )
                    break
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

            json_output = extract_json_from_response(message.choices[0].message.content)

            if json_output is not None:
                if (
                    not isinstance(json_output, dict)
                    or "content" not in json_output
                    or "output" not in json_output
                ):
                    print(f"Invalid schema format in output for row {idx + 1}")
                    failed_generations += 1
                    continue

                # validate against schema
                is_valid, validated_output = validate_with_pydantic(
                    json_output["output"]
                )

                if is_valid:
                    # Convert Pydantic model to dict for JSON serialization
                    json_output["output"] = validated_output.model_dump()

                    output_filename = os.path.join(
                        output_dir, f"genomicssample{idx + 1:04d}.json"
                    )

                    try:
                        with open(output_filename, "w", encoding="utf-8") as f:
                            json.dump(json_output, f, indent=4, ensure_ascii=False)
                        print(f"Successfully saved output to {output_filename}")
                        successful_generations += 1
                    except Exception as e:
                        print(f"Error saving JSON to file: {e}")
                        failed_generations += 1
                else:
                    print(f"Pydantic validation failed for row {idx + 1}")
                    # for debugging later
                    debug_filename = os.path.join(
                        output_dir,
                        f"invalid_genomicssample{idx + 1:04d}.json",
                    )
                    with open(debug_filename, "w", encoding="utf-8") as f:
                        json.dump(json_output, f, indent=4, ensure_ascii=False)
                    failed_generations += 1
            else:
                print(
                    f"Skipping file save for row {idx + 1} due to JSON parsing failure"
                )
                print("Claude response:", message.content)
                failed_generations += 1

        except Exception as e:
            print(f"Error processing row {idx + 1}: {e}")
            failed_generations += 1
            continue

    print(
        f"Processing complete: {successful_generations} successful, {failed_generations} failed"
    )


if __name__ == "__main__":
    # Read the arguments from CLI
    args = parse_CLI_args()

    # load api key
    load_dotenv()
    BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY")

    # Load credentials from config file
    config = load_config()

    BEDROCK_MODEL = config[args.model_name]["model"]
    os.environ["AWS_REGION_NAME"] = config[args.model_name]["region"]
    folder_name = f"samples_{args.model_name}/"

    process_bootstrap_rows(
        BEDROCK_MODEL, "bootstrap.csv", folder_name, args.sample_size
    )
