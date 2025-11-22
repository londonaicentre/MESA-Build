import argparse
import os

from dotenv import load_dotenv

from genollama_assets.prompts import generate_system_prompt, generate_bootstrap_user_prompt
from claudedatagen.claudedatagen import run_bootstrap_file_generation


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

def main():
    # Read the arguments from CLI
    args = parse_CLI_args()
    # load api key
    load_dotenv()
    run_bootstrap_file_generation(generate_system_prompt("systemprompt_bootstrap.md"), generate_bootstrap_user_prompt, args.instruction, args.model_name, os.getenv("BEDROCK_API_KEY"))