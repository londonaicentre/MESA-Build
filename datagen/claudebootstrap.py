import argparse
import os
import sys

from dotenv import load_dotenv

from llm_assets.prompts import generate_system_prompt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import load_config
from utils.aws import bedrock_completion


def parse_CLI_args() -> argparse.Namespace:
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
        "-i",
        "--instruction",
        type=str,
        required=True,
        help="Tailor the bootstrap file output, e.g. point the batch at a type of test, a disease area, a particular proband pattern, a report style, or any other variable",
    )
    arguments = parser.parse_args()
    return arguments


def generate_user_prompt(instructions: str) -> str:
    ## CREATE USER PROMPT

    user_prompt = f"""Please now generate 20 rows according to the above instructions as a CSV file. These rows should {instructions}. While conforming to these instructions, please also ensure that rows are varied, and represent a range of different report types and styles."""

    return user_prompt


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

    message = bedrock_completion(BEDROCK_MODEL, generate_system_prompt("systemprompt_bootstrap.md"), generate_user_prompt(args.instruction))
    with open("bootstrap.csv", "w", newline="") as file:
        file.write(message.choices[0].message.content)

    