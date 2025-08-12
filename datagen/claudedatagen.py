from litellm import completion
import pandas as pd
import json
import re
import time
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from schema.genomicextractmodel import GenomicTestReport

# load api key
load_dotenv()
BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY")
os.environ["AWS_REGION_NAME"] = "us-east-1"
BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"


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


def process_bootstrap_rows(bootstrap_file, output_dir, examples_dir="examples"):
    """
    Process rows from bootstrap.csv and generate synthetic genomic reports
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

    for idx, row in df.iterrows():
        print(f"Processing row {idx + samples_exist + 1}/{len(df)}")

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

            time.sleep(0.5)

            message = completion(
                model=BEDROCK_MODEL,
                max_tokens=8192,
                temperature=0.001,
                messages=[
                    {"content": system_prompt, "role": "system"},
                    {"content": user_prompt, "role": "user"},
                ],
                api_key=os.environ["BEDROCK_API_KEY"],
            )

            json_output = extract_json_from_response(message.choices[0].message.content)

            if json_output is not None:
                if (
                    not isinstance(json_output, dict)
                    or "content" not in json_output
                    or "output" not in json_output
                ):
                    print(
                        f"Invalid schema format in output for row {idx + samples_exist + 1}"
                    )
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
                        output_dir, f"genomicssample{idx + samples_exist + 1:04d}.json"
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
                    print(
                        f"Pydantic validation failed for row {idx + samples_exist + 1}"
                    )
                    # for debugging later
                    debug_filename = os.path.join(
                        output_dir,
                        f"invalid_genomicssample{idx + samples_exist + 1:04d}.json",
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
    process_bootstrap_rows("bootstrap.csv", "samples_sonnet4/")
